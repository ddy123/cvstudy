#!/usr/bin/env python3
"""
多模态RAG系统示例
结合文本RAG和LLaVA图像理解
"""
import os
import time
import base64
import requests
from PIL import Image
import io
from typing import List, Dict, Any, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import torch

class MultimodalRAGSystem:
    def __init__(self, 
                 rag_device: str = "cuda:0",
                 llava_device: str = "cuda:1"):
        """
        初始化多模态RAG系统
        
        Args:
            rag_device: RAG系统使用的GPU
            llava_device: LLaVA系统使用的GPU
        """
        print("初始化多模态RAG系统...")
        
        # 1. 初始化文本RAG系统
        print("初始化文本RAG系统...")
        self.rag_device = rag_device
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 第一张GPU给RAG
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': rag_device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        self.vector_store = None
        self.index_path = "faiss_index"
        
        # 2. 初始化LLaVA系统
        print("初始化LLaVA系统...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 第二张GPU给LLaVA
        
        # 这里我们假设LLaVA通过API服务运行
        self.llava_api_url = "http://localhost:8000/generate"
        self.llava_device = llava_device
        
        # 或者直接加载模型（如果内存允许）
        self.llava_model = None
        self.llava_processor = None
        
        # 尝试直接加载LLaVA模型
        self._load_llava_model()
        
        print("多模态RAG系统初始化完成")
    
    def _load_llava_model(self):
        """
        加载LLaVA模型
        """
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            print("加载LLaVA模型...")
            model_id = "llava-hf/llava-1.6-vicuna-7b-hf"  # 较小的模型
            
            self.llava_processor = AutoProcessor.from_pretrained(model_id)
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.llava_device
            )
            
            print("LLaVA模型加载成功")
            
        except Exception as e:
            print(f"LLaVA模型加载失败，将使用API模式: {e}")
            self.llava_model = None
    
    def create_text_index(self, documents: List[Dict[str, Any]]):
        """
        创建文本向量索引
        
        Args:
            documents: 文档列表
        """
        print("创建文本向量索引...")
        
        # 转换为LangChain文档
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            ))
        
        # 分割文档
        splits = self.text_splitter.split_documents(langchain_docs)
        print(f"文档分割为 {len(splits)} 个块")
        
        # 创建向量存储
        self.vector_store = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # 保存索引
        self.vector_store.save_local(self.index_path)
        print(f"索引已保存到: {self.index_path}")
    
    def load_text_index(self):
        """
        加载文本向量索引
        """
        if os.path.exists(self.index_path):
            print(f"加载文本向量索引: {self.index_path}")
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("文本索引加载成功")
        else:
            print("未找到文本索引文件")
    
    def text_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        文本搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            搜索结果
        """
        if self.vector_store is None:
            print("请先加载文本索引")
            return []
        
        results = []
        docs = self.vector_store.similarity_search(query, k=k)
        
        for i, doc in enumerate(docs):
            results.append({
                'rank': i + 1,
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata
            })
        
        return results
    
    def process_image_local(self, image_path: str, question: str) -> str:
        """
        使用本地LLaVA模型处理图像
        
        Args:
            image_path: 图像路径
            question: 问题
            
        Returns:
            回答
        """
        if self.llava_model is None or self.llava_processor is None:
            return "LLaVA模型未加载"
        
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")
            
            # 准备输入
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            
            inputs = self.llava_processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.llava_device)
            
            # 生成回答
            with torch.no_grad():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            
            # 解码输出
            response = self.llava_processor.decode(
                output[0], 
                skip_special_tokens=True
            )
            
            # 提取助手回答
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"图像处理失败: {str(e)}"
    
    def process_image_api(self, image_path: str, question: str) -> str:
        """
        通过API处理图像
        
        Args:
            image_path: 图像路径
            question: 问题
            
        Returns:
            回答
        """
        try:
            # 读取图像并编码为base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 准备请求数据
            payload = {
                "image": image_data,
                "prompt": question,
                "max_tokens": 200
            }
            
            # 发送请求
            response = requests.post(
                self.llava_api_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "未收到有效响应")
            else:
                return f"API请求失败: {response.status_code}"
                
        except Exception as e:
            return f"API调用失败: {str(e)}"
    
    def process_image(self, image_path: str, question: str) -> str:
        """
        处理图像（自动选择本地或API模式）
        
        Args:
            image_path: 图像路径
            question: 问题
            
        Returns:
            回答
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return f"图像文件不存在: {image_path}"
        
        # 优先使用本地模型
        if self.llava_model is not None:
            print("使用本地LLaVA模型处理图像")
            return self.process_image_local(image_path, question)
        else:
            print("使用API模式处理图像")
            return self.process_image_api(image_path, question)
    
    def multimodal_query(self, 
                        text_query: str, 
                        image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        多模态查询
        
        Args:
            text_query: 文本查询
            image_path: 图像路径（可选）
            
        Returns:
            查询结果
        """
        result = {
            "text_query": text_query,
            "image_path": image_path,
            "text_results": [],
            "image_response": None,
            "combined_response": None
        }
        
        # 1. 文本搜索
        print("执行文本搜索...")
        text_results = self.text_search(text_query, k=3)
        result["text_results"] = text_results
        
        # 2. 图像处理（如果提供了图像）
        if image_path and os.path.exists(image_path):
            print("处理图像...")
            image_response = self.process_image(image_path, text_query)
            result["image_response"] = image_response
            
            # 3. 综合响应
            print("生成综合响应...")
            combined_response = self._combine_responses(text_results, image_response, text_query)
            result["combined_response"] = combined_response
        
        return result
    
    def _combine_responses(self, 
                          text_results: List[Dict[str, Any]], 
                          image_response: str,
                          query: str) -> str:
        """
        综合文本和图像响应
        
        Args:
            text_results: 文本搜索结果
            image_response: 图像处理结果
            query: 原始查询
            
        Returns:
            综合响应
        """
        # 构建综合响应
        response = f"针对您的问题 '{query}'，以下是综合分析：\n\n"
        
        # 添加文本分析
        if text_results:
            response += "📚 基于文档分析：\n"
            for i, result in enumerate(text_results):
                response += f"{i+1}. {result['content']}\n"
            response += "\n"
        
        # 添加图像分析
        if image_response and image_response != "LLaVA模型未加载":
            response += "🖼️ 基于图像分析：\n"
            response += f"{image_response}\n\n"
        
        # 总结
        response += "💡 总结："
        if text_results and image_response:
            response += "结合文本资料和图像内容，"
        elif text_results:
            response += "根据文本资料，"
        elif image_response:
            response += "根据图像分析，"
        
        response += "以上是相关信息。"
        
        return response

def create_sample_data():
    """
    创建示例数据
    """
    # 创建示例文本文件
    text_file = "sample_tech.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("""计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取高层次的理解。

深度学习模型如卷积神经网络（CNN）在图像分类、物体检测和图像分割任务中表现出色。

OpenCV是一个开源的计算机视觉库，包含数百种计算机视觉算法。

图像处理技术包括图像增强、滤波、边缘检测和形态学操作。

计算机视觉应用包括人脸识别、自动驾驶汽车、医学图像分析和工业检测。""")
    
    # 创建示例图像（如果需要，可以下载一个示例图像）
    # 这里我们创建一个简单的PIL图像作为示例
    image_file = "sample_image.png"
    img = Image.new('RGB', (300, 200), color='blue')
    
    # 在图像上添加一些文本
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # 使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 80), "示例图像", fill="white", font=font)
    draw.text((50, 110), "计算机视觉示例", fill="yellow", font=font)
    
    img.save(image_file)
    
    return text_file, image_file

def main():
    """主函数"""
    print("=" * 60)
    print("多模态RAG系统示例")
    print("=" * 60)
    
    # 1. 初始化系统
    print("\n初始化多模态RAG系统...")
    multimodal_system = MultimodalRAGSystem(
        rag_device="cuda:0",
        llava_device="cuda:1"
    )
    
    # 2. 创建示例数据
    print("\n创建示例数据...")
    text_file, image_file = create_sample_data()
    
    # 3. 加载文本数据并创建索引
    print("\n加载文本数据...")
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    documents = [{
        'content': content,
        'metadata': {'source': text_file, 'type': '技术文档'}
    }]
    
    multimodal_system.create_text_index(documents)
    
    # 4. 测试文本搜索
    print("\n" + "=" * 60)
    print("测试文本搜索")
    print("=" * 60)
    
    text_queries = [
        "什么是计算机视觉？",
        "深度学习在计算机视觉中有什么应用？",
        "OpenCV是什么？"
    ]
    
    for query in text_queries:
        print(f"\n查询: {query}")
        results = multimodal_system.text_search(query, k=2)
        
        for result in results:
            print(f"  结果 {result['rank']}:")
            print(f"    内容: {result['content']}")
            print(f"    来源: {result['metadata'].get('source', '未知')}")
    
    # 5. 测试图像处理
    print("\n" + "=" * 60)
    print("测试图像处理")
    print("=" * 60)
    
    if os.path.exists(image_file):
        image_questions = [
            "描述这张图片",
            "图片中有什么文字？",
            "这张图片与计算机视觉有什么关系？"
        ]
        
        for question in image_questions:
            print(f"\n问题: {question}")
            response = multimodal_system.process_image(image_file, question)
            print(f"回答: {response}")
    
    # 6. 测试多模态查询
    print("\n" + "=" * 60)
    print("测试多模态查询")
    print("=" * 60)
    
    multimodal_queries = [
        ("计算机视觉有哪些应用？", image_file),
        ("如何分析图像？", image_file),
        ("人工智能在图像处理中的作用", None)  # 仅文本查询
    ]
    
    for text_query, img_file in multimodal_queries:
        print(f"\n多模态查询: {text_query}")
        if img_file:
            print(f"图像文件: {img_file}")
        
        result = multimodal_system.multimodal_query(text_query, img_file)
        
        if result["combined_response"]:
            print(f"综合响应:\n{result['combined_response']}")
        elif result["text_results"]:
            print("文本搜索结果:")
            for text_result in result["text_results"]:
                print(f"  - {text_result['content'][:100]}...")
        
        print("-" * 40)
    
    # 7. 清理文件
    print("\n清理示例文件...")
    for file_path in [text_file, image_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除: {file_path}")
    
    print("\n示例完成！")

if __name__ == "__main__":
    main()