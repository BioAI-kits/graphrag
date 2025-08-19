import logging
from pathlib import Path
import pandas as pd
from graphrag.utils.api import (
    get_embedding_store,
    load_search_prompt,
)
from graphrag.utils.api import create_storage_from_config

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.embeddings import (
    entity_description_embedding,
    community_full_content_embedding,
)
from graphrag.query.factory import get_local_search_engine, get_basic_search_engine, get_global_search_engine, \
    get_drift_search_engine
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_communities, read_indexer_reports, \
    read_indexer_text_units, read_indexer_relationships, read_indexer_covariates, read_indexer_report_embeddings
from graphrag.utils.storage import load_table_from_storage, storage_has_table

from serve.configs import settings

logger = logging.getLogger(__name__)


async def load_context(root: Path, data_dir: Path | None = None):
    # ① 读配置
    cfg = load_config(root)          # settings.yaml 默认为 root/settings.yaml
    if data_dir:
        cfg.output.base_dir = str(data_dir.resolve())

    # ② 解析索引产物
    dataframe_dict = await resolve_output_files(
        config=cfg,
        output_list=[
            "community_reports",
            "text_units",
            "relationships",
            "entities",
            "communities",
        ],
        optional_list=["covariates"],
    )
    return cfg, dataframe_dict


async def resolve_output_files(
    config: GraphRagConfig,
    output_list: list[str],
    optional_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read indexing output files into a dict of DataFrames."""
    # v2.4: 直接用 config.output 构造存储实例
    storage_obj = create_storage_from_config(config.output)

    dataframe_dict: dict[str, pd.DataFrame | None] = {}

    # 必选文件
    for name in output_list:
        dataframe_dict[name] = await load_table_from_storage(name, storage_obj)

    # 可选文件：不存在时返回 None
    if optional_list:
        for name in optional_list:
            if await storage_has_table(name, storage_obj):
                dataframe_dict[name] = await load_table_from_storage(name, storage_obj)
            else:
                dataframe_dict[name] = None

    return dataframe_dict


async def load_local_search_engine(config: GraphRagConfig, data: dict[str, pd.DataFrame]):
    vector_store_args = {k: v.model_dump()
                         for k, v in config.vector_store.items()}

    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,  # ← 常量已迁到 graphrag.config.embeddings
    )

    final_entities = data['entities']
    community_level = settings.community_level
    final_covariates = data['covariates'] if data.get('covariates') is not None else []
    final_text_units: pd.DataFrame = data["text_units"]
    final_relationships: pd.DataFrame = data["relationships"]
    final_community_reports: pd.DataFrame = data["community_reports"]
    final_communities: pd.DataFrame = data["communities"]

    entities_ = read_indexer_entities(final_entities, final_communities, community_level)
    covariates_ = read_indexer_covariates(final_covariates) if final_covariates else []
    prompt = load_search_prompt(config.root_dir, config.local_search.prompt)

    search_engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(final_community_reports, final_communities, community_level),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities_,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,  # type: ignore
        response_type=settings.response_type,
        system_prompt=prompt,
    )
    return search_engine


async def load_global_search_engine(config: GraphRagConfig, data: dict[str, pd.DataFrame]):
    final_entities = data['entities']
    community_level = settings.community_level
    final_community_reports: pd.DataFrame = data["community_reports"]
    final_communities: pd.DataFrame = data["communities"]

    communities_ = read_indexer_communities(final_communities, final_community_reports)
    reports = read_indexer_reports(
        final_community_reports = final_community_reports,
        final_communities=final_communities,
        community_level=community_level,
        dynamic_community_selection=settings.dynamic_community_selection,
    )
    entities_ = read_indexer_entities(final_entities,final_communities, community_level=community_level)
    map_prompt = load_search_prompt(config.root_dir, config.global_search.map_prompt)
    reduce_prompt = load_search_prompt(
        config.root_dir, config.global_search.reduce_prompt
    )
    knowledge_prompt = load_search_prompt(
        config.root_dir, config.global_search.knowledge_prompt
    )

    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities_,
        communities=communities_,
        response_type="Multiple Paragraphs",
        dynamic_community_selection=settings.dynamic_community_selection,
        map_system_prompt=map_prompt,
        reduce_system_prompt=reduce_prompt,
        general_knowledge_inclusion_prompt=knowledge_prompt,
    )
    return search_engine


async def load_drift_search_engine(config: GraphRagConfig, data: dict[str, pd.DataFrame]):
    vector_store_args = {k: v.model_dump()
                         for k, v in config.vector_store.items()}

    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,  # ← 常量已迁到 graphrag.config.embeddings
    )

    full_content_embedding_store = get_embedding_store(
        config_args=vector_store_args,  # type: ignore
        embedding_name=community_full_content_embedding,
    )

    final_entities = data['entities']
    community_level = settings.community_level
    final_text_units: pd.DataFrame = data["text_units"]
    final_relationships: pd.DataFrame = data["relationships"]
    final_community_reports: pd.DataFrame = data["community_reports"]
    final_communities: pd.DataFrame = data["communities"]

    entities_ = read_indexer_entities(final_entities,final_communities, community_level=community_level)
    reports = read_indexer_reports(
        final_community_reports=final_community_reports,
        final_communities=final_communities,
        community_level=community_level,
        dynamic_community_selection=settings.dynamic_community_selection,
    )
    read_indexer_report_embeddings(reports, full_content_embedding_store)
    prompt = load_search_prompt(config.root_dir, config.drift_search.prompt)

    search_engine = get_drift_search_engine(
        config=config,
        reports=reports,
        text_units=read_indexer_text_units(final_text_units),
        entities=entities_,
        relationships=read_indexer_relationships(final_relationships),
        description_embedding_store=description_embedding_store,  # type: ignore
        response_type =settings.response_type,
        local_system_prompt=prompt,
    )

    return search_engine


async def load_basic_search_engine(config: GraphRagConfig, data: dict[str, pd.DataFrame]):
    vector_store_args = {k: v.model_dump()
                         for k, v in config.vector_store.items()}

    description_embedding_store = get_embedding_store(
        config_args=vector_store_args,
        embedding_name=entity_description_embedding,  # ← 常量已迁到 graphrag.config.embeddings
    )

    final_text_units: pd.DataFrame = data["text_units"]

    prompt = load_search_prompt(config.root_dir, config.basic_search.prompt)

    search_engine = get_basic_search_engine(
        config=config,
        text_units=read_indexer_text_units(final_text_units),
        text_unit_embeddings=description_embedding_store,
        system_prompt=prompt,
    )

    return search_engine