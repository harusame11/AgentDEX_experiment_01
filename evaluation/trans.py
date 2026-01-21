import json

input_file = "gaia_level1.json"
output_file = "gaia_level1.jsonl"

print("正在转换...")

# 1. 读取标准的 JSON 文件 (一次性加载到内存)
with open(input_file, 'r', encoding='utf-8') as f_in:
    data_list = json.load(f_in) # 此时 data_list 是一个 Python 列表

# 2. 写入 JSONL 文件 (逐行写入)
with open(output_file, 'w', encoding='utf-8') as f_out:
    for entry in data_list:
        # dump 转为字符串，然后手动加换行符
        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"转换成功！共转换 {len(data_list)} 行数据。")