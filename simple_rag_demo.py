"""
简单的RAG (Retrieval-Augmented Generation) 演示脚本
展示RAG的核心逻辑流程
"""

import numpy as np
from typing import List, Tuple
import json

class SimpleRAG:
    """简单的RAG系统实现"""
    
    def __init__(self):
        self.documents = []  # 存储文档
        self.embeddings = []  # 存储文档向量
        self.index = {}  # 文档索引
    
    def add_document(self, doc_id: str, content: str):
        """添加文档到知识库"""
        self.documents.append({
            'id': doc_id,
            'content': content
        })
        
        # 生成文档向量 (这里用简单的词频模拟)
        embedding = self._generate_embedding(content)
        self.embeddings.append(embedding)
        
        # 更新索引
        self.index[doc_id] = len(self.documents) - 1
        print(f"文档 {doc_id} 已添加到知识库")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本向量 (简化版本)"""
        # 实际项目中会使用预训练模型如OpenAI embeddings或sentence-transformers
        words = text.lower().split()
        vocab = set(words)
        
        # 创建简单的词袋向量
        vector = np.random.rand(100)  # 模拟100维向量
        return vector
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """检索相关文档"""
        if not self.documents:
            return []
        
        # 生成查询向量
        query_embedding = self._generate_embedding(query)
        
        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # 使用余弦相似度 (简化版本)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # 按相似度排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, similarity in similarities[:top_k]:
            doc = self.documents[i]
            results.append((doc['content'], similarity))
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """基于检索到的上下文生成回答"""
        # 实际项目中会调用大语言模型API (如OpenAI GPT)
        
        # 构造提示词
        prompt = f"""
        基于以下上下文信息回答问题：
        
        上下文：
        {' '.join(context)}
        
        问题：{query}
        
        回答：
        """
        
        # 模拟LLM生成的回答
        answer = f"根据检索到的信息，关于'{query}'的回答是：基于提供的上下文，这里是生成的答案。"
        
        return answer
    
    def query(self, question: str) -> str:
        """RAG完整流程：检索 + 生成"""
        print(f"\n问题：{question}")
        print("=" * 50)
        
        # 步骤1：检索相关文档
        print("1. 检索相关文档...")
        retrieved_docs = self.retrieve(question, top_k=3)
        
        if not retrieved_docs:
            return "抱歉，知识库中没有相关信息。"
        
        # 显示检索结果
        print("检索到的相关文档：")
        context_list = []
        for i, (content, score) in enumerate(retrieved_docs, 1):
            print(f"  {i}. 相似度: {score:.3f}")
            print(f"     内容: {content[:100]}...")
            context_list.append(content)
        
        # 步骤2：生成回答
        print("\n2. 生成回答...")
        answer = self.generate_answer(question, context_list)
        
        print(f"\n最终回答：{answer}")
        return answer


def main():
    """演示RAG系统的使用"""
    print("简单RAG系统演示")
    print("=" * 50)
    
    # 创建RAG实例
    rag = SimpleRAG()
    
    # 添加一些示例文档到知识库
    documents = [
        ("doc1", "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。"),
        ("doc2", "机器学习是人工智能的一个子集，通过算法让计算机从数据中学习模式。"),
        ("doc3", "深度学习使用神经网络来处理复杂的数据模式，是机器学习的一个分支。"),
        ("doc4", "自然语言处理专注于让计算机理解和生成人类语言。"),
        ("doc5", "RAG技术结合了信息检索和文本生成，能够基于外部知识生成准确回答。")
    ]
    
    print("添加文档到知识库...")
    for doc_id, content in documents:
        rag.add_document(doc_id, content)
    
    print(f"\n知识库已包含 {len(documents)} 个文档")
    
    # 测试查询
    test_queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么关系？",
        "RAG技术是什么？"
    ]
    
    for query in test_queries:
        rag.query(query)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()


"""
实际RAG系统的改进方向：

1. 向量生成：
   - 使用预训练模型 (OpenAI embeddings, sentence-transformers)
   - 支持多语言向量化
   - 优化向量维度和质量

2. 检索优化：
   - 使用专业向量数据库 (Pinecone, Chroma, Weaviate)
   - 实现混合检索 (密集检索 + 稀疏检索)
   - 添加重排序机制

3. 文档处理：
   - 智能文档分割
   - 支持多种格式 (PDF, Word, HTML)
   - 文档预处理和清洗

4. 生成优化：
   - 集成大语言模型 (GPT-4, Claude, 本地模型)
   - 优化提示词工程
   - 实现流式生成

5. 评估和优化：
   - 检索质量评估
   - 生成质量评估
   - A/B测试和持续优化
"""