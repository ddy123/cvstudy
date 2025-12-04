"""
最小化香水原料向量检索系统
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
#from langchain_community.vectorstores import FAISS

# 1. 准备原料数据（简化版）
ingredients = [
    {
        "name": "玫瑰精油",
        "description": "从玫瑰花中提取的天然香料，甜美花香",
        "category": "花香"
    },
    {
        "name": "乙醇",
        "description": "香水中的溶剂，帮助挥发",
        "category": "溶剂"
    },
    {
        "name": "茉莉净油",
        "description": "茉莉花提取的浓郁花香",
        "category": "花香"
    }
]

# 2. 创建向量数据库
model = SentenceTransformer("all-MiniLM-L6-v2")  # 小模型，速度快

# 准备文本
texts = [f"{ing['name']} {ing['description']}" for ing in ingredients]

# 向量嵌入
embeddings = model.encode(texts, normalize_embeddings=True)

# 创建FAISS索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

print(f"✅ 创建了包含 {index.ntotal} 个向量的数据库")

# 3. 搜索函数
def search_ingredient(query, top_k=3):
    #print('开始搜索')
    """搜索原料"""
    # 将查询转换为向量
    query_vector = model.encode([query], normalize_embeddings=True)[0].astype('float32')
    
    # 搜索
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)
    
    # 处理结果
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= 0:
            ingredient = ingredients[idx]
            similarity = 1.0 / (1.0 + distance)  # 距离转相似度
            
            results.append({
                "name": ingredient["name"],
                "description": ingredient["description"],
                "category": ingredient["category"],
                "similarity": float(similarity)
            })
    
    return results

# 4. 测试搜索
print("\n🔍 测试搜索:")
test_queries = ["玫瑰", "酒精", "花香原料"]

for query in test_queries:
    print(f"\n搜索: '{query}'")
    results = search_ingredient(query)
    
    for result in results:
        print(f"  {result['name']} (相似度: {result['similarity']:.3f})")