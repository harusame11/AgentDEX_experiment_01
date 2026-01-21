from LLM_API import silicon_client
import faiss
import numpy as np
from openai import OpenAI
from datasets import load_dataset


# 加载语料
corpus = load_dataset('json', data_files='eval.jsonl', split='train')

# 批量 embedding
model_name = "Qwen/Qwen3-Embedding-8B"
batch_size = 100
all_embeddings = []

for i in range(0, len(corpus), batch_size):
    batch_texts = corpus[i:i+batch_size]['content']
    response = silicon_client.embeddings.create(
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
faiss.write_index(index, "eval.index")