import os
import time

import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost"  # or neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "admin@1234"
NEO4J_DATABASE = "neo4j"

# Create a Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session(database=NEO4J_DATABASE) as session:
    session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))

GRAPHRAG_FOLDER = os.path.join("../graphrag_zh", "output")


def batched_import(statement, df, batch_size=1000):
    """
    Import a dataframe into Neo4j using a batched approach.
    Parameters: statement is the Cypher query to execute, df is the dataframe to import, and batch_size is the number of rows to import in each batch.
    """
    total = len(df)
    start_s = time.time()
    for start in range(0, total, batch_size):
        batch = df.iloc[start: min(start + batch_size, total)]
        result = driver.execute_query("UNWIND $rows AS value " + statement,
                                      rows=batch.to_dict('records'),
                                      database_=NEO4J_DATABASE)
        print(result.summary.counters)
    print(f'{total} rows in {time.time() - start_s} s.')
    return total


# create constraints, idempotent operation

statements = """
create constraint chunk_id if not exists for (c:__Chunk__) require c.id is unique;
create constraint document_id if not exists for (d:__Document__) require d.id is unique;
create constraint entity_id if not exists for (c:__Community__) require c.community is unique;
create constraint entity_id if not exists for (e:__Entity__) require e.id is unique;
create constraint entity_title if not exists for (e:__Entity__) require e.name is unique;
create constraint entity_title if not exists for (e:__Covariate__) require e.title is unique;
create constraint related_id if not exists for ()-[rel:RELATED]->() require rel.id is unique;
""".split(";")

for statement in statements:
    if len((statement or "").strip()) > 0:
        print(statement)
        driver.execute_query(statement)

doc_df = pd.read_parquet(os.path.join(GRAPHRAG_FOLDER, "documents.parquet"), columns=["id", "title"])
doc_df.head(2)

# import documents
statement = """
MERGE (d:__Document__ {id:value.id})
SET d += value {.title}
"""

batched_import(statement, doc_df)

text_df = pd.read_parquet(os.path.join(GRAPHRAG_FOLDER, "text_units.parquet"),
                          columns=["id", "text", "n_tokens", "document_ids"])
text_df.head(2)

statement = """
MERGE (c:__Chunk__ {id:value.id})
SET c += value {.text, .n_tokens}
WITH c, value
UNWIND value.document_ids AS document
MATCH (d:__Document__ {id:document})
MERGE (c)-[:PART_OF]->(d)
"""

batched_import(statement, text_df)

tmp = pd.read_parquet(os.path.join(GRAPHRAG_FOLDER, "entities.parquet"))
tmp.head(2)

entity_df = (
    tmp
    .loc[:, [
        "id",                   # 必须存在
        "title",                 # 上面 rename 之后保证存在
        "type",
        "description",
        "human_readable_id",
        "text_unit_ids"
        # description_embedding 若确实没有，就先不读
    ]]
)


entity_statement = """
MERGE (e:__Entity__ {id:value.id})
SET   e += value {.human_readable_id, .description, .name}
WITH  e, value
// 如果 description_embedding 现在不存在，就先跳过向量写入
// CALL db.create.setNodeVectorProperty(e, "description_embedding", value.description_embedding)

CALL apoc.create.addLabels(
      e,
      CASE WHEN coalesce(value.type, "") = "" 
           THEN [] 
           ELSE [apoc.text.upperCamelCase(value.type)] 
      END) YIELD node
UNWIND value.text_unit_ids AS text_unit
MATCH (c:__Chunk__ {id: text_unit})
MERGE (c)-[:HAS_ENTITY]->(e)
"""

batched_import(entity_statement, entity_df)

rel_df = (
    pd.read_parquet(os.path.join(GRAPHRAG_FOLDER, "relationships.parquet"))
    # 如需沿用 rank 名称就重命名；否则直接用 combined_degree
    .rename(columns={"combined_degree": "rank"})
    .loc[:, [
        "source", "target", "id", "rank",
        "weight", "human_readable_id", "description", "text_unit_ids"
    ]]
)
rel_df.head(2)

rel_statement = """
MATCH (source:__Entity__ {name: replace(value.source, '"', '')})
MATCH (target:__Entity__ {name: replace(value.target, '"', '')})
MERGE (source)-[rel:RELATED {id: value.id}]->(target)
SET   rel.rank              = value.rank,             // 刚才 rename 过
      rel.weight            = value.weight,
      rel.human_readable_id = value.human_readable_id,
      rel.description       = value.description,
      rel.text_unit_ids     = value.text_unit_ids
RETURN count(*) AS createdRels
"""

batched_import(rel_statement, rel_df)

community_df = pd.read_parquet(os.path.join(GRAPHRAG_FOLDER, "communities.parquet"),
                               columns=["id", "level", "title", "text_unit_ids", "relationship_ids"])

community_df.head(2)

statement = """
MERGE (c:__Community__ {community:value.id})
SET c += value {.level, .title}
/*
UNWIND value.text_unit_ids as text_unit_id
MATCH (t:__Chunk__ {id:text_unit_id})
MERGE (c)-[:HAS_CHUNK]->(t)
WITH distinct c, value
*/
WITH *
UNWIND value.relationship_ids as rel_id
MATCH (start:__Entity__)-[:RELATED {id:rel_id}]->(end:__Entity__)
MERGE (start)-[:IN_COMMUNITY]->(c)
MERGE (end)-[:IN_COMMUNITY]->(c)
RETURn count(distinct c) as createdCommunities
"""

batched_import(statement, community_df)

cr_df = pd.read_parquet(
    os.path.join(GRAPHRAG_FOLDER, "community_reports.parquet")
)

# 2) （可选）把 rating_explanation ➜ rank_explanation，保持旧脚本不动
cr_df = cr_df.rename(columns={"rating_explanation": "rank_explanation"})

# 3) 按导入需要挑列；想全写入就保留全部
cols = ["id", "human_readable_id", "community", "level", "parent",
        "children", "title", "summary", "full_content", "rank",
        "rank_explanation",      # ← 如果你做了 rename
        "findings", "full_content_json", "period", "size"]
community_report_df = cr_df[cols]
community_report_df.head(2)

# community_df['findings'][0]

# import communities
community_statement = """
// 1️⃣ 基本属性写入
MERGE (c:__Community__ {community:value.community})
SET   c.id                 = value.id,
      c.human_readable_id  = value.human_readable_id,
      c.level              = value.level,
      c.title              = value.title,
      c.summary            = value.summary,
      c.full_content       = value.full_content,
      c.rank               = value.rank,
      c.rank_explanation   = value.rank_explanation,
      c.full_content_json  = value.full_content_json,
      c.period             = value.period,
      c.size               = value.size

// 2️⃣ 处理父子层级（可选）
WITH c, value
FOREACH (p IN CASE WHEN value.parent IS NULL THEN [] ELSE [value.parent] END |
  MERGE (pC:__Community__ {community:p})
  MERGE (c)-[:SUB_OF]->(pC)
)
FOREACH (ch IN value.children |
  MERGE (chC:__Community__ {community:ch})
  MERGE (chC)-[:SUB_OF]->(c)
)

// 3️⃣ 写入 findings（与旧脚本一致）
WITH c, value
UNWIND range(0, size(value.findings)-1) AS idx
WITH c, value, idx, value.findings[idx] AS f
MERGE (c)-[:HAS_FINDING]->(fin:Finding {id: idx})
SET   fin += f
"""
batched_import(community_statement, community_report_df)

# cov_df = pd.read_parquet(f'{GRAPHRAG_FOLDER}/create_final_covariates.parquet'),
# #                         columns=["id","text_unit_id"])
# cov_df.head(2)
# # Subject id do not match entity ids
#
#
# # import covariates
# cov_statement = """
# MERGE (c:__Covariate__ {id:value.id})
# SET c += apoc.map.clean(value, ["text_unit_id", "document_ids", "n_tokens"], [NULL, ""])
# WITH c, value
# MATCH (ch:__Chunk__ {id: value.text_unit_id})
# MERGE (ch)-[:HAS_COVARIATE]->(c)
# """
# batched_import(cov_statement, cov_df)