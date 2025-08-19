import pandas as pd


# 删除grade_label中的这些序号 并且model_name为 THUDM@GLM-4-32B-0414_label_jsons
grade_json_path = "grade_label/all_results.jsonl"
df_grade = pd.read_json(grade_json_path, lines=True)
new_grade = []
for index, row in df_grade.iterrows():
    if row['model_name'] == "THUDM@GLM-4-32B-0414_label_jsons":
        continue
    new_grade.append(row.to_dict())

new_df_grade = pd.DataFrame(new_grade)
new_df_grade.to_json(grade_json_path, orient='records', lines=True, force_ascii=False)