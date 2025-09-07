#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG API测试脚本
用于测试.env文件中配置的Anthropic API是否能够成功调用
"""

import os
import sys
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic

def load_environment():
    """加载环境变量"""
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    base_url = os.getenv('ANTHROPIC_BASE_URL')
    
    if not api_key:
        print("❌ 错误: 未找到ANTHROPIC_API_KEY环境变量")
        return None, None
    
    if not base_url:
        print("❌ 错误: 未找到ANTHROPIC_BASE_URL环境变量")
        return None, None
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"✅ Base URL: {base_url}")
    
    return api_key, base_url

def test_api_connection(api_key, base_url):
    """测试API连接"""
    try:
        print("\n🔄 正在测试API连接...")
        
        # 创建Anthropic客户端
        client = Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        
        # 发送一个简单的测试请求
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "请简单回复'API测试成功'，并说明你是Claude的哪个版本。"
                }
            ]
        )
        
        print("✅ API调用成功!")
        print(f"📝 响应内容: {response.content[0].text}")
        print(f"🔧 使用模型: {response.model}")
        print(f"📊 输入tokens: {response.usage.input_tokens}")
        print(f"📊 输出tokens: {response.usage.output_tokens}")
        
        return True
        
    except anthropic.AuthenticationError as e:
        print(f"❌ 认证错误: {e}")
        print("请检查API Key是否正确")
        return False
        
    except anthropic.APIConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("请检查Base URL是否正确或网络连接")
        return False
        
    except anthropic.RateLimitError as e:
        print(f"❌ 速率限制错误: {e}")
        print("请稍后重试")
        return False
        
    except anthropic.APIError as e:
        print(f"❌ API错误: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_rag_scenario(api_key, base_url):
    """测试RAG场景下的API调用"""
    try:
        print("\n🔄 正在测试RAG场景...")
        
        client = Anthropic(
            api_key=api_key,
            base_url=base_url
        )
        
        # 模拟RAG场景：提供上下文信息并提问
        context = """
        上下文信息：
        RAG（Retrieval-Augmented Generation）是一种结合了信息检索和生成式AI的技术。
        它通过检索相关文档片段，然后将这些信息作为上下文提供给大语言模型来生成更准确的回答。
        """
        
        question = "根据上述上下文，请解释什么是RAG技术？"
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\n问题：{question}"
                }
            ]
        )
        
        print("✅ RAG场景测试成功!")
        print(f"📝 RAG回答: {response.content[0].text}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG场景测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始RAG API测试")
    print("=" * 50)
    
    # 加载环境变量
    api_key, base_url = load_environment()
    if not api_key or not base_url:
        sys.exit(1)
    
    # 测试基本API连接
    basic_test_success = test_api_connection(api_key, base_url)
    
    if basic_test_success:
        # 测试RAG场景
        rag_test_success = test_rag_scenario(api_key, base_url)
        
        if rag_test_success:
            print("\n🎉 所有测试通过！API配置正确，可以用于RAG应用。")
        else:
            print("\n⚠️ 基本API测试通过，但RAG场景测试失败。")
    else:
        print("\n❌ API测试失败，请检查配置。")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ 测试完成")

if __name__ == "__main__":
    main()