PROMPT = """
<|im_start|>system
你是一名严格遵循 Rubric 进行学术问答评估的专家评审。请根据 Rubric 对“回答”进行逐项打分，并返回指定 JSON 格式，不得输出除 JSON 之外的任何内容。<|im_end|>
<|im_start|>user
{
  "rubric": {
    "scales": {
      "accuracy":   {"0": "完全错误/缺失", "5": "基本正确", "10": "完全正确"},
      "coverage":   {"0": "遗漏关键点", "5": "覆盖要点但不全", "10": "要点齐全"},
      "depth":      {"0": "无解释", "5": "一般性解释", "10": "深入机理与跨学科联系"},
      "traceability": {"0": "无引用", "5": "引用不全/格式差", "10": "引用清晰可查"},
      "clarity":    {"0": "用语混乱", "5": "尚可", "10": "严谨流畅"}
    }
  }
}
<|im_end|>

评分算法（隐含逻辑，供底层实现参考）逐项评分：每个二级指标打 0–10 分。
结果 JSON（仅示例）：
  {
      "accuracy": 8,
      "coverage": 9,
      "depth": 6,
      "traceability": 5,
      "clarity": 8,
      "comments": "回答准确，覆盖全面，但深度不足，引用不清晰。"
  }
提示必须严格输出 JSON；评测系统会自动解析。
comments 用中文，简明扼要指出主要优缺点。
"""

Tools = [
    {
        "type": "function",
        "function": {
            "name": "grade_answer",
            "description": "按照 Rubric 对模型回答进行多维度评分并输出 JSON 结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "accuracy": {"type": "number"},
                    "coverage": {"type": "number"},
                    "depth": {"type": "number"},
                    "traceability": {"type": "number"},
                    "clarity": {"type": "number"},
                    "comments": {"type": "string"}
                },
                "required": ["accuracy", "coverage", "depth", "traceability", "clarity", "comments"]
            }
        }
    }
]
