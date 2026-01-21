#!/usr/bin/env python3
"""
测试 API Embedding 模式
Test script for API embedding mode in retrieval_API.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from LLM_API import silicon_client
import numpy as np

def test_silicon_embedding(query:str):
    try:
        



if __name__ == "__main__":
    query = 'pyhton是什么？'
    success = test_silicon_embedding(query)
   

    if success:
        print("\n✅ 可以使用 API 模式启动 retrieval_API.py:")
        print("   python retrieval_API.py --use_api --port 8001")
    else:
        print("\n❌ 请修复 API 配置后再试")
        sys.exit(1)
