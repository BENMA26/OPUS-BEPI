import os
import pickle
import re

# ========== 参数设置 ==========
pdb_dir = "/work/home/maben/project/epitope_prediction/GraphBepi/data/BCE_633/PDB"   # ⚠️ 修改为你的目录路径
output_path = os.path.join(pdb_dir, "date.pkl")

# ========== 提取日期函数 ==========
def extract_date_from_pdb(file_path):
    """从pdb文件HEADER行提取日期，返回 [day, month, year]"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("HEADER"):
                # 匹配类似 01-MAY-13 的模式
                match = re.search(r"(\d{2})-([A-Z]{3})-(\d{2})", line)
                if match:
                    day, month, year = match.groups()
                    return [int(day), month, int(year)]
                break
    return None  # 没找到日期

# ========== 主逻辑 ==========
date_dict = {}

for filename in os.listdir(pdb_dir):
    if filename.lower().endswith(".pdb"):
        pdb_id = os.path.splitext(filename)[0]
        pdb_path = os.path.join(pdb_dir, filename)
        date_info = extract_date_from_pdb(pdb_path)
        if date_info:
            date_dict[pdb_id] = date_info

# ========== 保存为 pickle ==========
with open(output_path, "wb") as f:
    pickle.dump(date_dict, f)

print(date_dict)
print(f"✅ 已保存 {len(date_dict)} 个 PDB 文件的日期信息到 {output_path}")
