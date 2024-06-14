from qanything_kernel.configs.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING, OCR_MODEL_PATH
from typing import List
import time
from qanything_kernel.connector.llm.llm_for_openai_api import OpenAILLM
from langchain.schema import Document
from qanything_kernel.connector.database.mysql.mysql_client import KnowledgeBaseManager
from qanything_kernel.connector.database.faiss.faiss_client import FaissClient
from qanything_kernel.connector.rerank.rerank_backend import RerankBackend
from qanything_kernel.connector.embedding.embedding_backend import EmbeddingBackend
from qanything_kernel.utils.custom_log import debug_logger, qa_logger
from qanything_kernel.core.tools.web_search_tool import duckduckgo_search
from qanything_kernel.dependent_server.ocr_server.ocr import OCRQAnything
from qanything_kernel.utils.general_utils import num_tokens
from .local_file import LocalFile
import traceback
import base64
import numpy as np
import platform
import cv2
import asyncio
import json
import requests


class LocalDocQA:
    def __init__(self):
        self.llm: object = None
        self.embeddings: EmbeddingBackend = None
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.faiss_client: FaissClient = None
        self.mysql_client: KnowledgeBaseManager = None
        self.local_rerank_backend: RerankBackend = None
        self.ocr_reader: OCRQAnything = None
        self.mode: str = None
        self.use_cpu: bool = True
        self.model: str = None

    def get_ocr_result(self, input: dict):
        img_file = input['img64']
        height = input['height']
        width = input['width']
        channels = input['channels']
        binary_data = base64.b64decode(img_file)
        img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
        ocr_res = self.ocr_reader(img_array)
        res = [line for line in ocr_res if line]
        return res

    def init_cfg(self, args=None):
        self.rerank_top_k = int(args.model_size[0])
        self.use_cpu = args.use_cpu
        if args.use_openai_api:
            self.model = args.openai_api_model_name
        else:
            self.model = args.model.split('/')[-1]
        if platform.system() == 'Linux':
            if args.use_openai_api:
                self.llm: OpenAILLM = OpenAILLM(args)
            else:
                from qanything_kernel.connector.llm.llm_for_fastchat import OpenAICustomLLM
                self.llm: OpenAICustomLLM = OpenAICustomLLM(args)
            from qanything_kernel.connector.rerank.rerank_onnx_backend import RerankOnnxBackend
            from qanything_kernel.connector.embedding.embedding_onnx_backend import EmbeddingOnnxBackend
            self.local_rerank_backend: RerankOnnxBackend = RerankOnnxBackend(self.use_cpu)
            self.embeddings: EmbeddingOnnxBackend = EmbeddingOnnxBackend(self.use_cpu)
        else:
            if args.use_openai_api:
                self.llm: OpenAILLM = OpenAILLM(args)
            else:
                from qanything_kernel.connector.llm.llm_for_llamacpp import LlamaCPPCustomLLM
                self.llm: LlamaCPPCustomLLM = LlamaCPPCustomLLM(args)
            from qanything_kernel.connector.rerank.rerank_torch_backend import RerankTorchBackend
            from qanything_kernel.connector.embedding.embedding_torch_backend import EmbeddingTorchBackend
            self.local_rerank_backend: RerankTorchBackend = RerankTorchBackend(self.use_cpu)
            self.embeddings: EmbeddingTorchBackend = EmbeddingTorchBackend(self.use_cpu)
        self.mysql_client = KnowledgeBaseManager()
        self.ocr_reader = OCRQAnything(model_dir=OCR_MODEL_PATH, device="cpu")  # 省显存
        debug_logger.info(f"OCR DEVICE: {self.ocr_reader.device}")
        self.faiss_client = FaissClient(self.mysql_client, self.embeddings)

    async def insert_files_to_faiss(self, user_id, kb_id, local_files: List[LocalFile]):
        debug_logger.info(f'insert_files_to_faiss: {kb_id}')
        success_list = []
        failed_list = []

        for local_file in local_files:
            start = time.time()
            try:
                local_file.split_file_to_docs(self.get_ocr_result)
                if local_file.file_name.endswith('.faq'):
                    content_length = len(local_file.docs[0].metadata['faq_dict']['question']) + len(local_file.docs[0].metadata['faq_dict']['answer'])
                else:
                    content_length = sum([len(doc.page_content) for doc in local_file.docs])
            except Exception as e:
                error_info = f'split error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self.mysql_client.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.mysql_client.update_content_length(local_file.file_id, content_length)
            debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
            self.mysql_client.update_chunk_size(local_file.file_id, len(local_file.docs))
            add_ids = await self.faiss_client.add_document(local_file.docs)
            insert_time = time.time()
            debug_logger.info(f'insert time: {insert_time - end}')
            self.mysql_client.update_file_status(local_file.file_id, status='green')
            success_list.append(local_file)

            # for doc in local_file.docs:
            #     prompt = f"{doc.page_content} \n\n---\n你现在是一个文本摘要生成器，你要用超级超级精简的话，用中文提炼上面这段文字的要点成一句问题，一定不要啰嗦，一定要精简！一定要用中文！提炼的要点一定是一个能易于回答的问题。"
            #     # nonstream_answer = ""
            #     # async for reply in self.llm.generatorAnswer(prompt,history=[],streaming=False):
            #     # # return reply.llm_output['answer']
            #     #     nonstream_answer += reply.history[-1][1]
            #     history = ""
            #     async for answer_result in self.llm.generatorAnswer(prompt=prompt,
            #                                                 history=[],
            #                                                 streaming=False):
            #         resp = answer_result.llm_output["answer"]
            #         prompt = answer_result.prompt
            #         if not resp[6:].startswith("[DONE]"):
            #             history += answer_result.history[-1][1]

            #         response = {"history": history,
            #                     "prompt": prompt,
            #                     "result": resp,
            #         }
            #         # debug_logger.info(f"answer_result.llm_output: {answer_result.llm_output}\n answer_result.prompt: {answer_result.prompt}\n answer_result.history: {answer_result.history}")
            #     debug_logger.info(f"answer_result.llm_output: {answer_result.llm_output}\n answer_result.prompt: {answer_result.prompt}\n answer_result.history: {answer_result.history}")

            #     debug_logger.info(f"final prompt: {prompt}")
            #     debug_logger.info(f"final response: {response}")
                                  
                
        debug_logger.info(
            f"insert_to_faiss: success num: {len(success_list)}, failed num: {len(failed_list)}")

    def deduplicate_documents(self, source_docs):
        unique_docs = set()
        deduplicated_docs = []
        for doc in source_docs:
            if doc.page_content not in unique_docs:
                unique_docs.add(doc.page_content)
                deduplicated_docs.append(doc)
        return deduplicated_docs

    async def local_doc_search(self, query, kb_ids):
        source_documents = await self.get_source_documents(query, kb_ids)
        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        debug_logger.info(f"local doc search retrieval_documents: {retrieval_documents}")
        return retrieval_documents

    def get_web_search(self, queries, top_k=None):
        if not top_k:
            top_k = self.top_k
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query)
        source_documents = []
        for doc in web_documents:
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            source_documents.append(doc)
        return web_content, source_documents



    def web_page_search(self, query, top_k=None):
        # 防止get_web_search调用失败，需要try catch
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {e}")
            return []

        return source_documents


    async def get_source_documents(self, query, kb_ids, cosine_thresh=None, top_k=None):
        if not top_k:
            top_k = self.top_k
        source_documents = []
        t1 = time.time()
        filter = lambda metadata: metadata['kb_id'] in kb_ids
        # filter = None
        debug_logger.info(f"query: {query}")
        docs = await self.faiss_client.search(kb_ids, query, filter=filter, top_k=top_k)
        debug_logger.info(f"query_docs: {len(docs)}")
        t2 = time.time()
        debug_logger.info(f"faiss search time: {t2 - t1}")
        for idx, doc in enumerate(docs):
            if doc.metadata['file_name'].endswith('.faq'):
                faq_dict = doc.metadata['faq_dict']
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                nos_keys = faq_dict.get('nos_keys')
                doc.metadata['nos_keys'] = nos_keys
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            doc.metadata['embed_version'] = self.embeddings.getModelVersion
            source_documents.append(doc)
        if cosine_thresh:
            source_documents = [item for item in source_documents if float(item.metadata['score']) > cosine_thresh]

        return source_documents

    def reprocess_source_documents(self, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> List[Document]:
        # 组装prompt,根据max_token
        query_token_num = self.llm.num_tokens_from_messages([query])
        history_token_num = self.llm.num_tokens_from_messages([x for sublist in history for x in sublist])
        template_token_num = self.llm.num_tokens_from_messages([prompt_template])

        limited_token_nums = self.llm.token_window - self.llm.max_token - self.llm.offcut_token - query_token_num - history_token_num - template_token_num
        new_source_docs = []
        total_token_num = 0
        for doc in source_docs:
            doc_token_num = self.llm.num_tokens_from_docs([doc])
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                remaining_token_num = limited_token_nums - total_token_num
                doc_content = doc.page_content
                doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                while doc_content_token_num > remaining_token_num:
                    # Truncate the doc content to fit the remaining tokens
                    if len(doc_content) > 2 * self.llm.truncate_len:
                        doc_content = doc_content[self.llm.truncate_len: -self.llm.truncate_len]
                    else:  # 如果最后不够truncate_len长度的2倍，说明不够切了，直接赋值为空
                        doc_content = ""
                        break
                    doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                doc.page_content = doc_content
                new_source_docs.append(doc)
                break

        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"new_source_docs token nums: {self.llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs

    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        if num_tokens(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents

        scores = self.local_rerank_backend.get_rerank(query, [doc.page_content for doc in source_documents])
        debug_logger.info(f"rerank scores: {scores}")
        for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        return source_documents

    async def retrieve(self, query, kb_ids, need_web_search=False, score_threshold=0.35):
        retrieval_documents = await self.local_doc_search(query, kb_ids)
        if need_web_search:
            retrieval_documents.extend(self.web_page_search(query, top_k=3))
        debug_logger.info(f"retrieval_documents: {retrieval_documents}")
        if len(retrieval_documents) > 1:
            retrieval_documents = self.rerank_documents(query, retrieval_documents)
            # 删除掉分数低于阈值的文档
            tmp_documents = [item for item in retrieval_documents if float(item.metadata['score']) > score_threshold]
            if tmp_documents:
                retrieval_documents = tmp_documents

        # retrieval_documents = retrieval_documents[: self.rerank_top_k]

        debug_logger.info(f"reranked retrieval_documents: {retrieval_documents}")
        return retrieval_documents

    async def get_knowledge_based_answer(self, custom_prompt, query, kb_ids, chat_history=None, 
                                         streaming: bool = STREAMING,
                                         rerank: bool = False,
                                         need_web_search: bool = False):
        if chat_history is None:
            chat_history = []

        #retrieval_queries = [query]
        retrieval_documents = await self.retrieve(query, kb_ids, need_web_search=need_web_search)

        if custom_prompt is None:
            prompt_template = PROMPT_TEMPLATE
        else:
            prompt_template = custom_prompt + '\n' + PROMPT_TEMPLATE

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=prompt_template)
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)
        t1 = time.time()
        async for answer_result in self.llm.generatorAnswer(prompt=prompt,
                                                      history=chat_history,
                                                      streaming=streaming):
            resp = answer_result.llm_output["answer"]
            prompt = answer_result.prompt
            history = answer_result.history

            history[-1][0] = query
            response = {"query": query,
                        "prompt": prompt,
                        "result": resp,
                        "retrieval_documents": retrieval_documents,
                        "source_documents": source_documents}
            yield response, history
        t2 = time.time()
        debug_logger.info(f"LLM time: {t2 - t1}")



    async def generate_faq_questions(self, user_id, kb_id, local_files: List[LocalFile]):
        debug_logger.info(f'generate_faq_question: {kb_id}')
        success_list = []
        failed_list = []

        questions = []
        for local_file in local_files:
            start = time.time()
            try:
                local_file.split_file_to_docs(self.get_ocr_result)
                if local_file.file_name.endswith('.faq'):
                    content_length = len(local_file.docs[0].metadata['faq_dict']['question']) + len(local_file.docs[0].metadata['faq_dict']['answer'])
                else:
                    content_length = sum([len(doc.page_content) for doc in local_file.docs])
            except Exception as e:
                error_info = f'split error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self.mysql_client.update_file_status(local_file.file_id, status='red')
                failed_list.append(local_file)
                continue
            end = time.time()
            self.mysql_client.update_content_length(local_file.file_id, content_length)
            debug_logger.info(f'split time: {end - start} {len(local_file.docs)}')
            self.mysql_client.update_chunk_size(local_file.file_id, len(local_file.docs))
            add_ids = await self.faiss_client.add_document(local_file.docs)
            insert_time = time.time()
            debug_logger.info(f'insert time: {insert_time - end}')
            self.mysql_client.update_file_status(local_file.file_id, status='green')
            success_list.append(local_file)

            for doc in local_file.docs:
                # prompt = f"{doc.page_content} \n\n---\n你现在是一个文本摘要生成器，你要用超级超级精简的话，用中文提炼上面这段文字的要点并改写成一个问句形式的问题，一定不要啰嗦，一定要精简！一定要用中文！提炼的要点一定是一个能易于回答的问题。"
                prompt = f"{doc.page_content} \n\n---\n你现在是一个文本摘要生成器，你要用超级超级精简的话，用中文提炼上面这段文字的要点成一句问题，一定不要啰嗦，一定要精简！一定要用中文！提炼的要点一定是一个能易于回答的问题。"
                # nonstream_answer = ""
                # async for reply in self.llm.generatorAnswer(prompt,history=[],streaming=False):
                # # return reply.llm_output['answer']
                #     nonstream_answer += reply.history[-1][1]
                history = ""
                async for answer_result in self.llm.generatorAnswer(prompt=prompt,
                                                            history=[],
                                                            streaming=False):
                    resp = answer_result.llm_output["answer"]
                    prompt = answer_result.prompt
                    if not resp[6:].startswith("[DONE]"):
                        history += answer_result.history[-1][1]

                    response = {"history": history,
                                "prompt": prompt,
                                "result": resp,
                    }
                    await asyncio.sleep(5)
                #     # debug_logger.info(f"answer_result.llm_output: {answer_result.llm_output}\n answer_result.prompt: {answer_result.prompt}\n answer_result.history: {answer_result.history}")
                # debug_logger.info(f"answer_result.llm_output: {answer_result.llm_output}\n answer_result.prompt: {answer_result.prompt}\n answer_result.history: {answer_result.history}")

                # debug_logger.info(f"final prompt: {prompt}")
                # debug_logger.info(f"final response: {history}")
                questions.append(history)
                                  
        # file_ids = [file.file_id for file in local_files]
        # debug_logger.info(f"file_ids: {file_ids}")
        # self.faiss_client.delete_documents(kb_id,file_ids)
        # self.mysql_client.delete_files(kb_id,file_ids)
        debug_logger.info(
            f"generate_faq_question: success num: {len(questions)}")

        custom_prompt = None
        answers = []
        count = 0
        for question in questions:
            async for resp, history in self.get_knowledge_based_answer(custom_prompt=custom_prompt,
                                                                        query=question, 
                                                                        kb_ids=kb_id, 
                                                                        chat_history=[], 
                                                                        streaming=False, 
                                                                        rerank=True,
                                                                        need_web_search=False):            
                pass
            # await history

            count +=1
            debug_logger.info(f"get_knowledge_based_answer result: {history[-1][0]}")
            debug_logger.info(f"get_knowledge_based_answer history: {history[-1][-1]}")
            debug_logger.info(f"get_knowledge_based_answer count: {count}")
            answers.append(history[-1][-1])
            send_faq_request(question=question,answer=history[-1][-1],kb_id=kb_id)
            await asyncio.sleep(5)
            
        # if exist_file_names:
        #     msg = f'warning，当前的mode是soft，无法上传同名文件{exist_file_names}，如果想强制上传同名文件，请设置mode：strong'
        # else:
        #     msg = "success，后台正在飞速上传文件，请耐心等待"

        msg = "success，后台正在生成问答对，请耐心等待"

        file_ids = [file.file_id for file in local_files]
        debug_logger.info(f"file_ids: {file_ids}")
        self.faiss_client.delete_documents(kb_id,file_ids)
        self.mysql_client.delete_files(kb_id,file_ids)
        return questions


def send_faq_request(question,answer,kb_id):
    url = 'http://0.0.0.0:8777/api/local_doc_qa/upload_faqs'
    user_id = 'zzp'
    headers = {
        'content-type': 'application/json'
    }
    

    faqs = [
        {
            'question':question,
            'answer':answer,
            'nos_key':""
        }
    ]

    data = {
        'user_id': user_id,
        'kb_id': kb_id+'_FAQ',
        'faqs': faqs
    }
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=1)
        end_time = time.time()
        # response_times.append(end_time - start_time)
        res = response.json()
        print(res['response'])
        print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
    except Exception as e:
        print(f"请求发送失败: {e}")


