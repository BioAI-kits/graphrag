import re
from collections import defaultdict
from typing import Set, Dict

from serve.configs import settings

# ① 先把整段 […] 捕获下来（容忍有/无 ^、中英 Data/数据）
# ① 捕获外层 […]，可匹配 “Data:” 或 “数据:”
BRACKET_RE = re.compile(
    r'\[\s*\^?\s*(?:Data|数据)\s*:\s*(.*?)\]',  # .*? 只吃到右方括号
    re.I | re.S                                 # 不区分大小写，允许跨行
)

# ② 捕获 “Sources ( … ) / 实体 ( … )” 等，括号里先整段吃下
INNER_RE = re.compile(
    r'([A-Za-z\u4e00-\u9fa5]+)\s*\(([^)]*?)\)',  # [^)]*? → 直到遇到第一个 “)”
    re.I
)

def get_reference(text: str) -> Dict[str, Set[str]]:
    """返回形如 {'sources': {'257', '167'}, 'relationships': {'4768', ...}} 的字典"""
    data_dict: Dict[str, Set[str]] = defaultdict(set)

    for inside in BRACKET_RE.findall(text):
        # 分号通常用来分隔不同数据集，先粗切一刀
        for segment in inside.split(';'):
            for m in INNER_RE.finditer(segment):
                key = m.group(1).strip().lower()       # 统一小写
                ids = re.findall(r'\d+', m.group(2))    # 只保留纯数字
                data_dict[key].update(ids)
    return data_dict



def generate_ref_links(data: Dict[str, Set[str]], index_id: str) -> str:
    """
    把 ‘数据: 实体 (237)’ 这类脚注转成真正可点击的链接。
    每条单独生成一个脚注，方便前端渲染。
    """

    base_root = settings.website_address.rstrip("/")
    base_url = f"{base_root}/v1/references/{index_id}"

    key_map = {  # 中文 → 英文键映射
        "源": "sources", "来源": "sources",
        "关系": "relationships",
        "实体": "entities",
        "报告": "reports",
        "数据": "claims", "数据集": "claims",
    }

    # 1. 逐条拼成 Markdown 列表行
    items = []
    for key, ids in data.items():
        key_en = key_map.get(key, key)  # 映射或保留原样
        for rid in sorted(ids, key=lambda x: int(x) if x.isdigit() else x):
            url = f"{base_url}/{key_en}/{rid}"
            items.append(f"- [{key_en}:{rid}]({url})")

    # 2. 没有引用时返回空串；否则加标题
    if not items:
        return ""

    md = "\n".join(items)
    return md
