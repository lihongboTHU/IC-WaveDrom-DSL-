import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import numpy as np

# ==========================================
# 1. 数据解析与特征提取
# ==========================================
label_dir = "train_labels"

# 统计容器
categories = []
tokens = Counter()
protocols = []
complexities = []
seq_lengths = []
num_signals_list = []
data_validity = {"Hex Data": 0, "Empty Data (Noise)": 0}

print("🔍 正在扫描数据集...")
valid_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

for filename in valid_files:
    filepath = os.path.join(label_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            signals = data.get("signal", [])
        except:
            continue
            
    if not signals: continue
    
    names = [sig.get("name", "") for sig in signals]
    seq_len = len(signals[0].get("wave", ""))
    num_sig = len(signals)
    
    seq_lengths.append(seq_len)
    num_signals_list.append(num_sig)
    
    # 难度/复杂度计算：信号路数 * 周期数 + 包含的数据块数量
    complexity_score = (num_sig * seq_len) 
    complexities.append(complexity_score)
    
    # 协议推断
    if any("SDA" in n for n in names): protocols.append("I2C Bus")
    elif any("MOSI" in n for n in names): protocols.append("SPI Bus")
    elif any("ADDR" in n for n in names): protocols.append("Memory R/W")
    else: protocols.append("Mixed/Custom")

    for sig in signals:
        name = sig.get("name", "")
        wave = sig.get("wave", "")
        data_list = sig.get("data", [])
        
        # 统计字符
        tokens.update(list(wave))
        
        # 统计类别
        if any(k in name for k in ["CLK", "SCK", "SCL", "TCK"]):
            categories.append("Clock (CLK)")
        elif any(k in name for k in ["DAT", "MOSI", "MISO", "SDA", "RX", "TX", "DQ", "D0", "ADDR"]):
            categories.append("Data Bus")
        else:
            categories.append("Control/Other")
            
        # 统计 Data 真实性
        for d in data_list:
            if d.strip() == "": data_validity["Empty Data (Noise)"] += 1
            else: data_validity["Hex Data"] += 1

print(f"✅ 成功分析 {len(valid_files)} 条数据，正在生成可视化图表...")

# ==========================================
# 2. 绘制可视化大屏 (Dashboard)
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Timing Waveform Dataset Structural Analysis (N={len(valid_files)})', fontsize=22, fontweight='bold', y=0.98)

# --- 图1: 信号分类比例 (证明类别平衡) ---
cat_counts = pd.Series(categories).value_counts()
axes[0, 0].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', 
               colors=sns.color_palette("Set2"), startangle=140, wedgeprops={'edgecolor': 'white'})
axes[0, 0].set_title('1. Signal Category Balance', fontweight='bold')

# --- 图2: 物理协议覆盖率 (证明应用广泛性) ---
prot_counts = pd.Series(protocols).value_counts()
axes[0, 1].pie(prot_counts, labels=prot_counts.index, autopct='%1.1f%%', 
               colors=sns.color_palette("pastel"), startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
axes[0, 1].set_title('2. Hardware Protocol Coverage', fontweight='bold')

# --- 图3: 词表长尾分布 (证明特征丰富度) ---
tokens_df = pd.DataFrame(tokens.items(), columns=['Token', 'Count']).sort_values(by='Count', ascending=False)
sns.barplot(data=tokens_df, x='Token', y='Count', ax=axes[0, 2], palette="viridis")
axes[0, 2].set_yscale("log")
axes[0, 2].set_title('3. Wave Token Frequency (Log Scale)', fontweight='bold')

# --- 图4: 结构复杂度热力图 (证明空间多样性) ---
heatmap_data = pd.DataFrame({'Length': seq_lengths, 'Signals': num_signals_list})
pivot_table = heatmap_data.groupby(['Signals', 'Length']).size().unstack(fill_value=0)
sns.heatmap(pivot_table, cmap="YlGnBu", ax=axes[1, 0], cbar_kws={'label': 'Image Count'})
axes[1, 0].set_title('4. Structural Dimension Matrix', fontweight='bold')
axes[1, 0].set_xlabel('Total Cycles (Time Axis)')
axes[1, 0].set_ylabel('Number of Signals (Rows)')

# --- 图5: 标签难度分布 (证明课程学习梯度) - 改用绝对阈值 ---
easy_threshold = 30
hard_threshold = 120

sns.histplot(complexities, kde=True, color="crimson", bins=30, ax=axes[1, 1])
axes[1, 1].set_title('5. Structural Difficulty Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Complexity Score (Area + Density)')
axes[1, 1].set_ylabel('Frequency')

# 标出难度区间的绝对边界（竖虚线）
axes[1, 1].axvline(easy_threshold, color='gray', linestyle='--', linewidth=1.5)
axes[1, 1].axvline(hard_threshold, color='gray', linestyle='--', linewidth=1.5)

# 动态计算文字 Y 坐标
y_top = axes[1, 1].get_ylim()[1] * 0.9

# 获取当前 X 轴范围
x_min, x_max = axes[1, 1].get_xlim()

# 为 Easy、Medium、Hard 选择合理的 X 坐标（基于绝对阈值）
# Easy: 放在 easy_threshold 左侧一半距离，但不能超出左边界
easy_x = max(x_min, easy_threshold / 2) if easy_threshold / 2 > x_min else easy_threshold - (easy_threshold - x_min) / 2
axes[1, 1].text(easy_x, y_top, 'Easy', ha='center', color='black', fontweight='bold')

# Medium: 放在 easy_threshold 和 hard_threshold 的中点
medium_x = (easy_threshold + hard_threshold) / 2
if medium_x < x_min:
    medium_x = x_min + (x_max - x_min) * 0.3
elif medium_x > x_max:
    medium_x = x_max - (x_max - x_min) * 0.3
axes[1, 1].text(medium_x, y_top, 'Medium', ha='center', color='black', fontweight='bold')

# Hard: 放在 hard_threshold 右侧一定比例处，例如到右端点的1/3位置
hard_x = min(x_max, hard_threshold + (x_max - hard_threshold) * 0.3)
if hard_x <= hard_threshold:
    hard_x = hard_threshold + (x_max - hard_threshold) * 0.1
axes[1, 1].text(hard_x, y_top, 'Hard', ha='center', color='black', fontweight='bold')

# --- 图6: 有效数据密度 ---
axes[1, 2].bar(data_validity.keys(), data_validity.values(), color=['#4CAF50', '#FF9800'])
axes[1, 2].set_title('6. Payload Data Validity Ratio', fontweight='bold')
for i, v in enumerate(data_validity.values()):
    axes[1, 2].text(i, v + max(data_validity.values())*0.02, str(v), ha='center', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_file = 'dataset_scientific_analysis.png'
plt.savefig(output_file, dpi=300)
print(f"🎉 统计图表已生成完毕！请查看当前目录下的：{output_file}")