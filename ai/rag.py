#!/usr/bin/env python3
"""
基础RAG系统使用示例
演示文档加载、向量检索和LLM问答
"""
import os
import time
from typing import List, Dict, Any
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import VLLM

class BasicRAGSystem:
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-large-zh-v1.5",
                 llm_model: str = "Qwen/Qwen2.5-7B-Instruct",  # 或使用其他vLLM支持的模型
                 device: str = "cuda:0"):
        """
        初始化RAG系统
        
        Args:
            embedding_model: 嵌入模型名称
            llm_model: LLM模型名称
            device: 使用的GPU设备
        """
        print("初始化RAG系统...")
        
        # 设置环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
        
        # 1. 初始化嵌入模型
        print(f"加载嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 3. 初始化LLM（使用vLLM）
        print(f"加载LLM模型: {llm_model}")
        self.llm = VLLM(
            model=llm_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            trust_remote_code=True,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512
        )
        
        # 4. 初始化向量存储
        self.vector_store = None
        self.index_path = "faiss_index"
        
        print("RAG系统初始化完成")
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        加载文档
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            文档列表
        """
        documents = []
        
        for file_path in file_paths:
            print(f"加载文档: {file_path}")
            
            try:
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.lower().endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, encoding='utf-8')
                else:
                    print(f"不支持的文件格式: {file_path}")
                    continue
                
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append({
                        'content': doc.page_content,
                        'source': file_path,
                        'page': doc.metadata.get('page', 0)
                    })
                
                print(f"  加载了 {len(loaded_docs)} 个页面")
                
            except Exception as e:
                print(f"  加载失败: {str(e)}")
        
        return documents
    
    def create_index(self, documents: List[Dict[str, Any]]):
        """
        创建向量索引
        
        Args:
            documents: 文档列表
        """
        print("创建向量索引...")
        
        # 转换为LangChain文档格式
        from langchain.schema import Document
        langchain_docs = []
        
        for doc in documents:
            langchain_docs.append(Document(
                page_content=doc['content'],
                metadata={
                    'source': doc['source'],
                    'page': doc['page']
                }
            ))
        
        # 分割文档
        print("分割文档...")
        splits = self.text_splitter.split_documents(langchain_docs)
        print(f"文档分割为 {len(splits)} 个块")
        
        # 创建向量存储
        print("创建向量存储...")
        self.vector_store = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # 保存索引
        self.vector_store.save_local(self.index_path)
        print(f"索引已保存到: {self.index_path}")
    
    def load_index(self):
        """
        加载已有的向量索引
        """
        if os.path.exists(self.index_path):
            print(f"加载向量索引: {self.index_path}")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("索引加载成功")
        else:
            print("未找到索引文件，请先创建索引")
    
    def create_qa_chain(self):
        """
        创建问答链
        """
        if self.vector_store is None:
            print("请先加载或创建向量索引")
            return None
        
        # 自定义提示模板
        prompt_template = """基于以下上下文信息，回答用户的问题。如果你不知道答案，请诚实地说明你不知道，不要编造答案。

上下文信息:
{context}

问题: {question}

请提供详细、准确的回答，如果上下文中没有相关信息，请说明不知道:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 创建检索QA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            print("请先加载或创建向量索引")
            return []
        
        results = []
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        for i, (doc, score) in enumerate(docs_and_scores):
            results.append({
                'rank': i + 1,
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'source': doc.metadata.get('source', '未知'),
                'page': doc.metadata.get('page', '未知'),
                'score': float(score)
            })
        
        return results
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        回答问题
        
        Args:
            question: 问题
            
        Returns:
            回答结果
        """
        qa_chain = self.create_qa_chain()
        if qa_chain is None:
            return {"answer": "系统未初始化", "sources": []}
        
        print(f"提问: {question}")
        start_time = time.time()
        
        try:
            result = qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": [],
                "time": time.time() - start_time
            }
            
            # 提取来源信息
            for doc in result.get("source_documents", []):
                response["sources"].append({
                    "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "source": doc.metadata.get("source", "未知"),
                    "page": doc.metadata.get("page", "未知")
                })
            
            return response
            
        except Exception as e:
            return {
                "answer": f"回答问题出错: {str(e)}",
                "sources": [],
                "time": time.time() - start_time
            }

def create_sample_documents():
    """
    创建示例文档
    """
    sample_docs = []
    
    # 创建示例文本文件
    doc1_path = "sample_ai.txt"
    with open(doc1_path, "w", encoding="utf-8") as f:
        f.write("""人工智能（AI）是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。
        
人工智能的主要分支包括机器学习、深度学习、自然语言处理、计算机视觉和机器人技术。

机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下从数据中学习。

深度学习是机器学习的一个子集，使用神经网络架构，特别是深度神经网络。

自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。

人工智能已应用于各个领域，包括医疗诊断、自动驾驶汽车、推荐系统和金融预测。""")
    
    doc2_path = "sample_llm.txt"
    with open(doc2_path, "w", encoding="utf-8") as f:
        f.write("""大语言模型（LLM）是基于深度学习的自然语言处理模型，具有数十亿甚至数万亿参数。

著名的大语言模型包括GPT系列、BERT、T5、PaLM和LLaMA。

这些模型通过在大规模文本语料库上进行训练，学习语言的统计模式。

大语言模型可以执行各种任务，包括文本生成、翻译、摘要、问答和代码生成。

微调是一种技术，通过在小规模特定任务数据上进一步训练预训练模型，使其适应特定任务。

提示工程是通过精心设计输入提示来引导LLM产生期望输出的实践。""")
    
    return [doc1_path, doc2_path]

def main():
    """主函数"""
    print("=" * 60)
    print("纯文本RAG系统示例")
    print("=" * 60)
    
    # 1. 初始化RAG系统
    rag_system = BasicRAGSystem(
        embedding_model="BAAI/bge-large-zh-v1.5",
        llm_model="Qwen/Qwen2.5-7B-Instruct",  # 请确保vLLM支持此模型
        device="cuda:0"
    )
    
    # 2. 创建示例文档
    print("\n创建示例文档...")
    sample_files = create_sample_documents()
    
    # 3. 加载文档
    print("\n加载文档...")
    documents = rag_system.load_documents(sample_files)
    
    # 4. 创建索引
    print("\n创建向量索引...")
    rag_system.create_index(documents)
    
    # 5. 测试相似性搜索
    print("\n" + "=" * 60)
    print("测试相似性搜索")
    print("=" * 60)
    
    test_queries = [
        "什么是机器学习？",
        "大语言模型有哪些应用？",
        "深度学习是什么？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = rag_system.search_similar(query, k=2)
        
        for result in results:
            print(f"  相关文档 {result['rank']}:")
            print(f"    内容: {result['content']}")
            print(f"    来源: {result['source']}")
            print(f"    相似度: {result['score']:.4f}")
    
    # 6. 测试问答
    print("\n" + "=" * 60)
    print("测试问答功能")
    print("=" * 60)
    
    test_questions = [
        "请解释什么是人工智能？",
        "大语言模型有什么用途？",
        "机器学习和深度学习有什么区别？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        response = rag_system.ask_question(question)
        
        print(f"回答: {response['answer']}")
        print(f"回答时间: {response['time']:.2f}秒")
        
        if response['sources']:
            print("来源:")
            for i, source in enumerate(response['sources']):
                print(f"  {i+1}. {source['source']} (第{source['page']}页)")
                print(f"     内容: {source['content']}")
    
    # 7. 清理示例文件
    print("\n清理示例文件...")
    for file_path in sample_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除: {file_path}")
    
    print("\n示例完成！")

if __name__ == "__main__":
    main()