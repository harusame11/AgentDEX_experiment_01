import json
import os
from datasets import load_dataset

HF_TOKEN =os.getenv("HF_TOKEN")

try:
    print("正在加载 GAIA Level 1 数据...")
    
    # 加载完整数据集
    dataset = load_dataset(
        "gaia-benchmark/GAIA", 
        "2023_level1", 
        split="validation", 
        token=HF_TOKEN,
        trust_remote_code=True
    )

    print(f"原始数据总共有: {len(dataset)} 条")
    print("开始清洗数据（移除包含 file_path 的条目）...")

    cleaned_data_list = []

    # 遍历所有数据
    for item in dataset:
        file_path = item.get("file_path", "")
        if file_path and str(file_path).strip() != "":
            continue 

        current_id_num = len(cleaned_data_list)
        
        # 3. 格式重组
        new_entry = {
            "id": f"gaia_levle1__{current_id_num}",  # 保持你之前的命名格式
            "question": item["Question"],
            "answer": item["Final answer"]
        }
        
        cleaned_data_list.append(new_entry)

    # 修改输出文件名，体现这是全量清洗后的数据
    output_file = "gaia_level1_cleaned_full.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data_list, f, ensure_ascii=False, indent=2)

   

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"发生错误: {e}")