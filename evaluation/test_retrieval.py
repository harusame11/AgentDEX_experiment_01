import requests
import json

# 1. 服务地址
URL = "http://127.0.0.1:8000/retrieve"

# 2. 准备请求数据
# ⚠️ 注意：由于服务端逻辑限制，'eid' 必须存在于服务端的 examples.json 中
# 如果你是刚开始测试，请确保服务端 examples.json 里至少有一个 key 为 "0" 的数据
payload = {
    # 这里的查询内容建议写一个和你语料库相关的
    "queries": ["什么是python？"],  
    
    # 返回多少条结果
    "topk": 5,
    
    # 是否返回相似度分数
    "return_scores": True,
    
    # 参数已经弃用：必须对应 examples.json 里的 Key
    "eid": "0"  
}

headers = {
    "Content-Type": "application/json"
}

def call_rag_service():
    print(f"🚀 正在请求 RAG 服务: {URL}")
    print(f"📋 参数: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(URL, json=payload, headers=headers)
        
        # 检查 HTTP 状态码
        if response.status_code == 200:
            results = response.json()
            
            # 服务端返回的是列表的列表 (batch search)
            # 结构: [ [ {doc}, {doc} ], [ {doc} ] ]
            if not results or not results[0]:
                print("\n⚠️  警告：服务返回了空结果！")
                print("可能原因：")
                print(f"1. 服务端的 examples.json 中没有配置 eid='{payload['eid']}' 的白名单。")
                print("2. 检索到的文档 ID 不在白名单列表里（被过滤掉了）。")
                print("3. 本地库没检索到，且 Tavily 联网搜索未配置 Key。")
            else:
                print(f"\n✅ 成功检索到 {len(results[0])} 条文档：")
                for idx, item in enumerate(results[0]):
                    score = item.get('score', 'N/A')
                    tavily_score = item.get('tavily_score', None)
                    content = item.get('document', {}).get('content', '')

                    # 判断来源
                    if isinstance(score, (int, float)) and score < 0:
                        source = f"🌐 网络搜索 | 排序: {score:.0f} | Tavily分数: {tavily_score:.4f}" if tavily_score else f"🌐 网络搜索 | 排序: {score:.0f}"
                    else:
                        source = f"📁 本地文档 | 相似度: {score:.4f}"

                    # 截取一部分内容展示
                    preview = content[:1000].replace('\n', ' ') + "..."
                    print(f"--- [结果 {idx+1}] {source} ---")
                    print(f"内容: {preview}\n")
        
        elif response.status_code == 500:
            print("\n❌ 服务器内部错误 (500)")
            print("极有可能是 'eid' 在服务端的 examples.json 中不存在，导致 KeyError。")
        
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ 无法连接到服务器。请确认服务是否已在 {URL} 启动。")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

if __name__ == "__main__":
    call_rag_service()