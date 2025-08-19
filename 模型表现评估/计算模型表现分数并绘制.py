import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
# 中文
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ===== 1. 读取 JSONL 文件 =====
file_path = "grade_label/all_results.jsonl"  # 👉 替换为你的实际文件路径
df = pd.read_json(file_path, lines=True)

# ===== 2. 展开 response 字段 =====
response_df = pd.json_normalize(df['response'])
df_expanded = pd.concat([df, response_df], axis=1)
df_expanded.drop(columns=['response'], inplace=True)


def map_difficulty_from_id(id_val):
    if 1 <= id_val <= 100:
        return 'easy'
    elif 101 <= id_val <= 200:
        return 'medium'
    elif 201 <= id_val <= 300:
        return 'hard'
    else:
        return 'unknown'

df_expanded['difficulty'] = df_expanded['id'].apply(map_difficulty_from_id)

# ===== 4. 定义加权策略并计算加权得分 =====
difficulty_weight_adjust = {
    "easy":   {"accuracy": 0.5, "coverage": 0.25, "depth": 0.1, "traceability": 0.05, "clarity": 0.1},
    "medium": {"accuracy": 0.45, "coverage": 0.2, "depth": 0.15, "traceability": 0.1, "clarity": 0.1},
    "hard":   {"accuracy": 0.4, "coverage": 0.15, "depth": 0.2, "traceability": 0.15, "clarity": 0.1}
}

def calculate_weighted_score(row):
    weights = difficulty_weight_adjust.get(row['difficulty'], {})
    return sum(row.get(k, 0) * w for k, w in weights.items())

# ===== 3. 重新标记 difficulty =====
df_expanded['id'] = df_expanded['id'].astype(int)
df_expanded["model_name"] = df_expanded["model_name"].astype(str).str.replace("_label_jsons", "", regex=False)
df_expanded['weighted_score'] = df_expanded.apply(calculate_weighted_score, axis=1)
print(df_expanded.head().to_string())
# ===== 5. 按模型和难度聚合加权总分 =====
grouped = df_expanded.groupby(['model_name', 'difficulty'])['weighted_score'].mean().reset_index()

# ===== 6. 绘制加权得分柱状图 =====
plt.figure(figsize=(15, 8))
sns.barplot(data=grouped, x='difficulty', y='weighted_score', hue='model_name')
plt.title("不同模型在不同难度下的加权总分表现")
plt.xlabel("难度等级")
plt.ylabel("加权平均总分")
plt.legend(title="模型名称", loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
# 保存
plt.savefig("model_performance_by_difficulty.png")

score_columns = ["accuracy", "coverage", "depth", "traceability", "clarity"]

os.makedirs("score_rankings", exist_ok=True)

for metric in score_columns:
    # 分组平均
    metric_df = df_expanded.groupby(['model_name', 'difficulty'])[metric].mean().reset_index()
    # 创建模型名 (难度) 组合标签
    metric_df['标签'] = metric_df['model_name'] + " (" + metric_df['difficulty'] + ")"
    # 按分数排序
    metric_df = metric_df.sort_values(by=metric, ascending=True)

    # 绘图
    plt.figure(figsize=(10, max(5, 0.4 * len(metric_df))))
    sns.barplot(data=metric_df, x=metric, y='标签', palette="Blues_d")
    plt.title(f"{metric}：各模型在不同难度下得分（由低到高）")
    plt.xlabel("平均得分")
    plt.ylabel("模型 (难度)")
    plt.tight_layout()
    plt.savefig(f"score_rankings/{metric}_score_ranking.png")