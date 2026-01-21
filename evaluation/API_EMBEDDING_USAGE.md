# API Embedding Mode for retrieval_API.py

## 概述 (Overview)

`retrieval_API.py` 现在支持两种工作模式:

1. **本地模式 (Local Mode)**: 使用本地 Qwen/Qwen3-Embedding-8B 模型 (需要 GPU)
2. **API 模式 (API Mode)**: 使用 SiliconFlow API embedding 服务 (无需 GPU)

## 修改内容 (Changes Made)

### 1. Encoder 类重构

- 新增 `use_api` 参数控制工作模式
- 拆分 `encode()` 方法为两个内部方法:
  - `_encode_via_api()`: 调用 SiliconFlow API
  - `_encode_via_local_model()`: 使用本地 PyTorch 模型

### 2. Config 类扩展

新增三个配置参数:
- `use_api`: bool - 是否使用 API 模式
- `api_model_name`: str - API embedding 模型名称
- `api_embedding_dim`: int - embedding 维度

### 3. 命令行参数

新增三个启动参数:
```bash
--use_api              # 启用 API 模式
--api_model_name       # 指定 API 模型 (默认: BAAI/bge-large-en-v1.5)
--api_embedding_dim    # 指定 embedding 维度 (默认: 1024)
```

## 使用方法 (Usage)

### 本地模式 (Local Mode - 默认)

需要 GPU 和本地 Qwen 模型:

```bash
export INDEX_DIR="/path/to/index"
python retrieval_API.py \
    --port 8001 \
    --new_cache_dir cache/hle \
    --example_id_file examples.json
```

### API 模式 (API Mode - 推荐)

**无需 GPU**, 使用 SiliconFlow API:

```bash
export INDEX_DIR="/path/to/index"
python retrieval_API.py \
    --use_api \
    --port 8001 \
    --new_cache_dir cache/hle \
    --example_id_file examples.json
```

使用自定义 API 模型:

```bash
python retrieval_API.py \
    --use_api \
    --api_model_name "BAAI/bge-m3" \
    --api_embedding_dim 1024 \
    --port 8001
```

## SiliconFlow 支持的 Embedding 模型

| 模型名称 | 维度 | 适用场景 |
|---------|------|---------|
| `BAAI/bge-large-en-v1.5` | 1024 | 英文检索 (推荐) |
| `BAAI/bge-m3` | 1024 | 多语言检索 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 中文检索 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 轻量级英文 |

## ⚠️ 重要注意事项

### 1. FAISS 索引兼容性

**API 模式的 embedding 维度必须与现有 FAISS 索引的维度匹配!**

检查现有索引维度:
```python
import faiss
index = faiss.read_index("eval.index")
print(f"Index dimension: {index.d}")
```

如果不匹配，需要:
1. 使用相同维度的 API 模型
2. 或重新构建 FAISS 索引

### 2. 重新构建索引 (使用 API Embedding)

如果需要用 API embedding 构建新索引:

```python
# build_index_api.py
import faiss
import numpy as np
from openai import OpenAI
from datasets import load_dataset

# 初始化 SiliconFlow 客户端
client = OpenAI(
    api_key="YOUR_SILICON_API_KEY",
    base_url="https://api.siliconflow.cn/v1"
)

# 加载语料
corpus = load_dataset('json', data_files='eval.jsonl', split='train')

# 批量 embedding
model_name = "BAAI/bge-large-en-v1.5"
batch_size = 100
all_embeddings = []

for i in range(0, len(corpus), batch_size):
    batch_texts = corpus[i:i+batch_size]['content']
    response = client.embeddings.create(
        model=model_name,
        input=batch_texts,
        encoding_format="float"
    )
    embeddings = [data.embedding for data in response.data]
    all_embeddings.extend(embeddings)

# 构建 FAISS 索引
embeddings_array = np.array(all_embeddings, dtype=np.float32)
# L2 归一化
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
embeddings_array = embeddings_array / norms

# 使用 Inner Product 索引 (归一化后等价于余弦相似度)
index = faiss.IndexFlatIP(embeddings_array.shape[1])
index.add(embeddings_array)

# 保存索引
faiss.write_index(index, "eval_api.index")
```

### 3. 性能对比

| 模式 | GPU 需求 | 速度 | 成本 | 适用场景 |
|-----|---------|------|------|---------|
| 本地模式 | 需要 (>= 16GB VRAM) | 快 | 硬件成本 | 大规模检索 |
| API 模式 | 不需要 | 中等 | API 调用费用 | 中小规模/开发测试 |

### 4. API 限流

SiliconFlow API 有调用频率限制，建议:
- 调整 `retrieval_batch_size` 参数
- 添加重试逻辑
- 监控 API quota

## 测试 API 模式

启动服务:
```bash
python retrieval_API.py --use_api --port 8001
```

测试检索:
```python
import requests

response = requests.post(
    "http://localhost:8001/retrieve",
    json={
        "queries": ["What is machine learning?"],
        "topk": 5,
        "return_scores": True,
        "eid": "test_001"
    }
)

results = response.json()
print(f"Found {len(results[0])} documents")
```

## Tavily 回退机制

无论使用哪种模式，当本地 RAG 返回结果 < 3 条时，系统会自动:
1. 调用 Tavily search API
2. 提取网页全文内容
3. 合并到返回结果

这个机制在两种模式下都正常工作。

## 故障排查

### 问题: "API Embedding Error: ..."

**解决方案**:
1. 检查 `LLM_API.py` 中 `SILICON_API_KEY` 是否正确
2. 确认 API quota 未耗尽
3. 检查网络连接

### 问题: FAISS 索引维度不匹配

**错误信息**: `RuntimeError: Error in virtual void faiss::IndexFlat::add(...)`

**解决方案**:
1. 确认索引维度: `index.d`
2. 使用相同维度的 API 模型
3. 或使用上面的脚本重新构建索引

### 问题: GPU 内存不足 (本地模式)

**解决方案**:
1. 切换到 API 模式: `--use_api`
2. 或减小 batch size: 修改 `retrieval_batch_size`

## 总结

✅ **推荐使用 API 模式** 的场景:
- 无 GPU 或 GPU 内存不足
- 开发和测试阶段
- 中小规模检索 (< 1000 queries/day)

✅ **推荐使用本地模式** 的场景:
- 有足够 GPU 资源
- 大规模生产环境
- 对延迟要求极高

API 模式让你可以在没有 GPU 的情况下快速开发和测试 RAG 系统!
