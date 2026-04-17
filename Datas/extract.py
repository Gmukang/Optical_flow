import pandas as pd

# ========== 仅需修改这3个参数 ==========
input_file = "melt.csv"          # 你的原始CSV文件
output_file = "melt_m.csv"       # 输出文件
target_ids = [0]            # 你要提取的Target_ID列表
# ======================================

# 读取数据
df = pd.read_csv(input_file, encoding="utf-8-sig")

# 1. 筛选目标ID的行
# 2. 只保留你指定的列
selected_columns = [
    "Frame",
    "Target_ID",
    "CX",
    "CY",
    "Area",
    "IoU_Similarity",
    "RD",
    "BMS",
    "Final_Result"
]

filtered_df = df[df["Target_ID"].isin(target_ids)][selected_columns]

# 保存结果
filtered_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"提取完成！共 {len(filtered_df)} 行数据")
print(f"已保留指定列，保存到：{output_file}")