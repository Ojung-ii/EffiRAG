import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
import torch

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.hyper_reranking_w_treerep import QueryGraphBuilder, HyperReranker, SubgraphConfig, HyperbolicEmbedder, StructralSimilarity
# from .utils.hyper_reranking import QueryGraphBuilder, HyperReranker, SubgraphConfig, HyperbolicEmbedder, StructralSimilarity
from .utils.save_utils import append_manifest_row, save_graph_with_ids, now_run_id
from .utils.adaptive_rerank import adaptive_rerank
logger = logging.getLogger(__name__)

class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 dataset = None,
                 struct_k=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        self.logger = logger
        
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint
        if struct_k is not None :
            self.global_config.struct_k = struct_k
        if dataset is not None :
            self.global_config.dataset = dataset

        self.all_debug_info = [] # ★★★ 디버그 정보 저장을 위한 리스트 초기화 ★★★
        
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False
        
    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None
    ) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking
        """

        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)
        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            query_id = self.make_query_id(query)
            run_id = now_run_id()   # 지금은 매 쿼리마다 새로 만들지 말고, run 시작 시 1번만 만들 것을 추천(아래 5단계)

            case_dir = self._case_dir(query_id, run_id)

            # fact score도 같이 저장 (candidate/topk)
            facts_payload = {
                "query": query,
                "q_idx": q_idx,
                "link_top_k": self.global_config.linking_top_k,
                "facts_before_rerank": rerank_log.get("facts_before_rerank", []),
                "facts_after_rerank": rerank_log.get("facts_after_rerank", []),
            }

            self._dump_json(case_dir / "01_filtered_triples.json", facts_payload)

            rerank_end = time.time()
            print(rerank_log)
            self.rerank_time += rerank_end - rerank_start

            # --- 쿼리 ID / 런 ID 준비 ------------------------------------------------
            query_id = getattr(self, "current_query_id", None) or getattr(self, "last_query_id", None)
            if not query_id:
                query_id = self.make_query_id(query)  # 문자열 해시 등
            run_id = now_run_id()

            dbg = None
            # --- ★ START: Reranking/QuerySolution 로직 대체 ★ -----------------------
            if len(top_k_facts) == 0:
                logger.info("No facts found after reranking, return DPR results")
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                reranked_doc_ids = sorted_doc_ids[:num_to_retrieve]
                reranked_scores = sorted_doc_scores[:num_to_retrieve]
            else:
                # 1. PPR 기반 초기 랭킹 (Structure Search)
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight,
                    debug_ctx={"query_id": query_id, "run_id": run_id, "q_idx": q_idx, "query": query}

                )

                # 2. Query Graph 생성
                query_graph = QueryGraphBuilder(
                    save_dir=f"./outputs/{self.global_config.dataset}/query_graph_analysis",
                    dataset_name=self.global_config.dataset,
                ).build(top_k_facts, query_id=query_id)

                
                case_dir = self._case_dir(query_id, run_id)
                qg_path = case_dir / "03_q_graph.graphml"
                query_graph.write_graphml(str(qg_path))

                self._dump_json(case_dir / "03_q_graph_meta.json", {
                    "nodes": query_graph.vcount(),
                    "edges": query_graph.ecount(),
                })
                
                # 3. Hyperbolic Reranking 실행 (External Call)
                cfg = SubgraphConfig(
                                            subgraph_mode="ppr",
                                            max_depth=2,
                                            per_hop_topk=12,
                                            allow_topk=150,            # 160 -> 150
                                            per_node_top_r=6,
                                            flow_quantile=0.85,
                                            rank_metric="ppr",
                                            keep_bridges=True,
                                            keep_articulation=True,
                                            keep_top_edge_betw=0,

                                                        )

                cfg_lorentz = SubgraphConfig(
                                                subgraph_mode="ppr",
                                                    max_depth=3,
                                                    per_hop_topk=12,
                                                    allow_topk=150,
                                                    per_node_top_r=6,
                                                    flow_quantile=0.85,
                                                    rank_metric="ppr",
                                                    keep_bridges=True,
                                                    keep_articulation=True,
                                                    keep_top_edge_betw=0,
                                                                )
                                                                
                reranker = HyperReranker(
                    g=self.graph,
                    save_dir=f"./outputs/{self.global_config.dataset}/subgraph_analysis",
                    dataset_name=self.global_config.dataset,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embedder = HyperbolicEmbedder(hr_instance=self, embed_dim=32, curvature=1.0, device=device, hgnn_type="hgat")
                scorer = StructralSimilarity(mode="cosine")
                from .utils.hyper_reranking_w_treerep import LTRConfig
                ALPHA_OPTIMAL = 0.8 
                LTR_DISABLED = LTRConfig(enabled=False)
                LTR_ENABLED_CONFIG = LTRConfig(enabled=True)
                TREEREP_BACKEND_SAFE = "bfs" 
                USE_NO_STATIC_FEAT = False # Static Feat는 제거되었으므로 False로 고정
                
                import numpy as np

                k_rrf_cand = 120
                pids = np.array(sorted_doc_ids[:num_to_retrieve])
                sem_ranks = np.arange(1, len(pids) + 1)

                raw_ppr = self._extract_ppr_for_pids(
                    pids=pids,
                    last_ppr_scores=getattr(self, "last_ppr_scores", None),
                    passage_node_idxs=getattr(self, "passage_node_idxs", None),
                    default_val=0.0,
                )
                ppr_order = np.argsort(-raw_ppr)
                ppr_ranks = np.empty_like(ppr_order); ppr_ranks[ppr_order] = np.arange(1, len(pids) + 1)

                rrf = 1.0/(k_rrf_cand + sem_ranks) + 1.0/(k_rrf_cand + ppr_ranks)
                order = np.argsort(-rrf)
                pre_ids    = pids[order].tolist()
                pre_scores = np.asarray(sorted_doc_scores[:num_to_retrieve])[order].tolist()

                reranked_doc_ids, reranked_scores, dbg = adaptive_rerank(
                    query_graph=query_graph,
                    passage_topk_ids=pre_ids,
                    passage_topk_scores=pre_scores,
                    passage_node_idxs=self.passage_node_idxs,
                    last_ppr_scores=getattr(self, "last_ppr_scores", None),
                    cfg=cfg_lorentz, embedder=embedder, scorer=scorer,
                    passage_embeddings=self.passage_embeddings, entity_embeddings=self.entity_embeddings,
                    reranker=reranker,
# --- 핵심 Rerank/Fusion 파라미터 ---
sim_cac_mode = "lorentz",
alpha = 0.65             ,  # 구조 기여도 (1 - alpha) = 0.35
gating_enabled = True     , # 🔥 게이팅 활성화 (ON)
gate_floor = 0.60,
gate_ceiling = 0.90,        # 또는 0.95 (안정화를 위한 설정)
lorentz_temp = 2.0,
k_rrf = 14,
pin_k = 2,
enable_hgnn_fusion = True,
attn_topk = 12            , # EXP-D에서 성능 향상에 기여했던 핵심 값
attn_lambda = 7.5,
attn_residual = 0.10,
treerep_backend = "bfs",
use_simple_struct_feats = True, # 구조 피처 (depth, radial) 통합 활성화
hgnn_type = "hgat" ,      # 또는 "hgat" (실험에 따라 다름)
# struct_k = 30            # 이전 로그에서 언급되었던 값 (추가적인 구조 필터링)
                )
                                
    #             reranked_doc_ids, reranked_scores = reranker.rerank_with_structure(
    #                 query_graph=query_graph,
    #                 passage_topk_ids=sorted_doc_ids[:num_to_retrieve],
    #                 passage_topk_scores=sorted_doc_scores[:num_to_retrieve],
    #                 passage_node_idxs=self.passage_node_idxs,
    #                 last_ppr_scores=getattr(self, "last_ppr_scores", None),
    #                 cfg=cfg,
    #                 embedder=embedder,
    #                 scorer=scorer,
    #                 passage_embeddings=self.passage_embeddings,
    #                 entity_embeddings=self.entity_embeddings,
    #                 # question_embedding=self.question_embedding,
    #                 # node2pid=self.node2pid,
    #                 # alpha=self.global_config.alpha,
    #                 # struct_k=self.global_config.struct_k,
    # struct_k=30,
    # sim_cac_mode="tangent",
    # alpha=0.72,                # 구조 기여 더↑
    # gating_enabled=True,
    # gate_floor=0.60,
    # gate_ceiling=0.88,
    # attn_topk=10,
    # attn_lambda=7.5,
    # attn_residual=0.10,        # 0.2 -> 0.10 (희석↓)
    # k_rrf=14, pin_k=0,
    # treerep_backend=TREEREP_BACKEND_SAFE,
    # ltr_config=LTR_DISABLED,
    # enable_hgnn_fusion=False,
    #             )

            try:
                mode = dbg['rerank_kwargs'].get('sim_cac_mode', 'N/A')
                alpha = dbg['rerank_kwargs'].get('alpha', 'N/A')
                struct_k = dbg['rerank_kwargs'].get('struct_k', 'N/A')
                
                # cfg 정보도 추가
                depth = dbg['cfg'].get('max_depth', 'N/A')
                topk = dbg['cfg'].get('per_hop_topk', 'N/A')

                self.logger.info(f"[ADAPTIVE] Mode={mode} | Alpha={alpha:.2f} | Struct_k={struct_k} | Subgraph={depth}d/{topk}k")
            except Exception as e:
                self.logger.warning(f"[ADAPTIVE] Error accessing debug info: {e}")

            # self.logger.info(f"[ADAPTIVE] mode={dbg['mode']} policy={dbg['policy']} feat={dbg['features']}")

            # 4. QuerySolution 생성 및 추가
            top_k_docs = [
                self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                for idx in reranked_doc_ids
            ]

        # ★★★ START: DBG 정보 수집 및 저장 로직 ★★★
            if dbg is not None:
                # rho_spearman과 p_value_spearman을 dbg에 저장했으므로,
                # 이를 포함한 dbg 딕셔너리 전체를 수집합니다.
                self.all_debug_info.append({
                    "query_id": query_id,
                    "q_idx": q_idx,
                    "dbg": dbg
                })


            retrieval_results.append(
                QuerySolution(
                    question=query,
                    docs=top_k_docs,
                    doc_scores=reranked_scores,
                )
            )

        # --- ★ END: Reranking/QuerySolution 로직 대체 ★ -----------------------------

        retrieve_end_time = time.time()
        self.all_retrieval_time += retrieve_end_time - retrieve_start_time
        self.misc_time = self.all_retrieval_time - (self.rerank_time + self.ppr_time)

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.misc_time:.2f}s")

        self._save_all_debug_info() # <--- 별도의 저장 함수를 호출합니다.
        
        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[r.docs for r in retrieval_results],
                k_list=k_list,
            )
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            retrieval_time_dict = {'retrieval_time': self.all_retrieval_time, 'ppr_time': self.ppr_time, 'rerank_time': self.rerank_time, 'misc_time': self.misc_time}
            logger.info("Total Evaluation Summary:------------------------------------------------")
            logger.info(f"<----run_mode")
            logger.info(f"Model: {self.global_config.llm_name}")
            logger.info(f"Embedding Model: {self.global_config.embedding_model_name}")
            logger.info(f"Dataset: {self.global_config.dataset}")
            # logger.info(f"sim_cac_mode: {getattr(self.global_config, 'sim_cac_mode', 'tangent')}")
            # logger.info(f"alpha: {self.global_config.alpha}")
            logger.info(f"Retrieval Time: {retrieval_time_dict}")
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            logger.info(f"Evaluation results for QA: {overall_qa_results}")
            
            return queries_solutions, all_response_message, all_metadata,retrieval_time_dict, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]
            retrieval_time_dict = {'retrieval_time': self.all_retrieval_time, 'ppr_time': self.ppr_time, 'rerank_time': self.rerank_time, 'misc_time': self.misc_time}
            return queries_solutions, all_response_message, all_metadata,retrieval_time_dict, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

        
        
    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05,
                                         debug_ctx: dict | None = None
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        if debug_ctx is not None:
            case_dir = self._case_dir(debug_ctx["query_id"], debug_ctx["run_id"])

            # node_weights는 길이가 그래프 전체 노드라서 무거움 → topN만 저장 추천
            topN = 200
            ppr = np.asarray(self.last_ppr_scores, dtype=float)
            top_nodes = np.argsort(-ppr)[:topN]

            ppr_payload = {
                "q_idx": debug_ctx["q_idx"],
                "query": debug_ctx["query"],
                "damping": self.global_config.damping,
                "linking_score_map_top30": linking_score_map,  # 이미 top30로 줄여둠
                "ppr_top_nodes": [
                    {"node_idx": int(i), "node_name": self.graph.vs[i]["name"], "ppr": float(ppr[i])}
                    for i in top_nodes
                ],
                "ppr_passage_ranking_top200": [
                    {"pid": int(pid), "score": float(sc)}
                    for pid, sc in zip(ppr_sorted_doc_ids[:200].tolist(), ppr_sorted_doc_scores[:200].tolist())
                ],
            }
            self._dump_json(case_dir / "02_ppr_search.json", ppr_payload)

                
        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        # ✅ 전체 노드용 PPR 점수 보관 (rerank_with_structure에서 사용)
        self.last_ppr_scores = np.asarray(pagerank_scores, dtype=float)

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

        
    def _get_question_embedding_for(self, q_idx: int) -> np.ndarray:
        """
        self.get_query_embeddings(queries) 가 만들어 둔 self.query_embeddings 를 1개 꺼내서 반환.
        형상이 (D,) 가 되도록 numpy float32 로 정규화.
        """
        vec = None
        # 1) 권장: self.get_query_embeddings 가 채워둔 결과 사용
        if hasattr(self, "query_embeddings") and self.query_embeddings is not None:
            # 보통 (N, D) or List[List[float]] 형태
            if isinstance(self.query_embeddings, (list, tuple)):
                vec = self.query_embeddings[q_idx]
            else:
                # numpy array (N, D)
                vec = self.query_embeddings[q_idx, :]

        # 2) 혹시 위가 비어있다면, 임시로 현재 쿼리를 임베딩해서 사용 (모델 보유 시)
        if vec is None:
            try:
                encoder = getattr(self, "query_encoder", None)
                if encoder is not None:
                    text = self.current_query_text if hasattr(self, "current_query_text") else ""
                    if hasattr(self, "_all_queries_cache"):
                        text = self._all_queries_cache[q_idx]
                    vec = encoder.encode([text])[0]
            except Exception:
                pass

        if vec is None:
            # 최후의 안전장치: 0벡터
            d = getattr(self, "embedding_dim", None)
            if d is None and hasattr(self, "passage_embeddings"):
                d = int(self.passage_embeddings.shape[1])
            if d is None:
                d = 768
            return np.zeros((d,), dtype=np.float32)

        return np.asarray(vec, dtype=np.float32).reshape(-1)

    import numpy as np

    @staticmethod
    def _extract_ppr_for_pids(
        pids,
        last_ppr_scores,
        passage_node_idxs=None,
        default_val: float = 0.0,
    ):
        import numpy as np
        n = len(pids)
        if last_ppr_scores is None:
            return np.full(n, default_val, dtype=float)

        if isinstance(last_ppr_scores, dict):
            if isinstance(passage_node_idxs, dict):
                out = []
                for pid in pids:
                    node_idx = passage_node_idxs.get(pid, None)
                    if node_idx is not None and node_idx in last_ppr_scores:
                        out.append(last_ppr_scores.get(node_idx, default_val))
                    else:
                        out.append(last_ppr_scores.get(pid, default_val))
                return np.asarray(out, dtype=float)
            return np.asarray([last_ppr_scores.get(pid, default_val) for pid in pids], dtype=float)

        if isinstance(last_ppr_scores, (list, tuple)) and len(last_ppr_scores) > 0 \
           and isinstance(last_ppr_scores[0], (list, tuple)) and len(last_ppr_scores[0]) == 2:
            d = {int(k): float(v) for (k, v) in last_ppr_scores}
            if isinstance(passage_node_idxs, dict):
                out = []
                for pid in pids:
                    node_idx = passage_node_idxs.get(pid, None)
                    out.append(float(d.get(int(node_idx), default_val)) if node_idx is not None else default_val)
                return np.asarray(out, dtype=float)
            return np.full(n, default_val, dtype=float)

        arr = np.asarray(last_ppr_scores)
        if arr.ndim == 1:
            if isinstance(passage_node_idxs, dict):
                idxs = np.array([passage_node_idxs.get(pid, -1) for pid in pids], dtype=int)
                out = np.full(n, default_val, dtype=float)
                mask = (idxs >= 0) & (idxs < arr.shape[0])
                out[mask] = arr[idxs[mask]]
                return out
            if arr.shape[0] == n:
                return arr.astype(float, copy=False)
            return np.full(n, default_val, dtype=float)

        return np.full(n, default_val, dtype=float)

    
    
    
    def _build_node2pid(self, graph: ig.Graph) -> Dict[str, int]:
        """
        그래프 노드명 -> passage_embeddings row index 매핑.
        관례:
        - name 이 'chunk-<int>' 이면 <int> 를 passage id 로 사용
        - vs attribute 'passage_id' 가 있으면 그 값을 사용
        다른 케이스는 매핑 생략(임베더가 자체 fallback 로직 보유)
        """
        node2pid: Dict[str, int] = {}
        names = graph.vs["name"] if "name" in graph.vs.attributes() else []
        # 1) 'passage_id' 속성이 있는 경우
        if "passage_id" in graph.vs.attributes():
            for name, pid in zip(names, graph.vs["passage_id"]):
                try:
                    pid_int = int(pid)
                    if pid_int >= 0:
                        node2pid[name] = pid_int
                except Exception:
                    continue

        # 2) 'chunk-<id>' 패턴인 경우
        for name in names:
            if isinstance(name, str) and name.startswith("chunk-"):
                try:
                    pid = int(name.split("-")[-1])
                    if pid >= 0:
                        node2pid[name] = pid
                except Exception:
                    continue

        return node2pid

    #_---------------------------
    # def rerank_with_structure(
    #     self,
    #     query_graph: Graph,
    #     semantic_top_k_ids: list[int],
    #     semantic_top_k_scores : list[float],
    #     alpha : float = 0.5,
    #     embed_dim: int = 10,
    #     curvature: float = 1.0,
    #     similarity_mode: str = "cosine",
    #     device: str = "cpu"
    # ) -> tuple[list[int], list[float]]:
    #     """
    #     구조 기반 하이퍼볼릭 유사도 계산 후 rerank
    #     """

    #     top_k_root_ids = semantic_top_k_ids
    #     tree_builder = TreeStructureSubgraph(self.graph)
    #     subgraphs = tree_builder.total_subgraph(top_k_root_ids)

    #     embedder = HyperbolicEmbedder(embed_dim=embed_dim, curvature=curvature, device=device)
    #     # query_lorentz = embedder.embed_lorentz(query_graph)
    #     # subgraph_lorentz_list = [embedder.embed_lorentz(g) for g in subgraphs]

    #     transported_query_list = embedder.transport_query_to_all(query_lorentz, subgraph_lorentz_list)
    #     projected_subgraph_list = embedder.logmap0_batch(subgraph_lorentz_list)

    #     # Hyperbolic similarity computation
    #     scorer = StructralSimilarity(mode=similarity_mode)
    #     structral_similarities = scorer.compute_listwise(transported_query_list, projected_subgraph_list)
    #     print(f"structral_similarities: {structral_similarities}")
    #     norm_semantic_top_k_scores = scorer.normalize(semantic_top_k_scores)
    #     print(f"norm_semantic_top_k_scores: {norm_semantic_top_k_scores}")
    #     norm_structral_similarities = scorer.normalize(structral_similarities)
    #     print(f"norm_structral_similarities: {norm_structral_similarities}")
    #     final_similarity_scores = alpha* (norm_semantic_top_k_scores) + (1 - alpha) * (norm_structral_similarities)

    #     # final similarities with the top_k_root_ids
    #     reranked = sorted(zip(top_k_root_ids, final_similarity_scores.tolist()), key=lambda x: x[1], reverse=True)
    #     reranked_doc_ids = [doc_id for doc_id, _ in reranked]
    #     reranked_scores = [score for _, score in reranked]

    #     return reranked_doc_ids, reranked_scores


    # 원점 기준으로 하이퍼볼릭 임베딩 ---------------------------
    # def rerank_with_structure(
    #     self,
    #     query_graph: ig.Graph,
    #     semantic_top_k_ids: list[int],
    #     semantic_top_k_scores: list[float],
    #     alpha: float = 0.7,
    #     embed_dim: int = 32,
    #     curvature: float = 1.0,
    #     similarity_mode: str = "cosine",
    #     struct_k : int = 30,
    #     device: str = "cpu",
    # ) -> tuple[list[int], list[float]]:
    #     """
    #     구조 기반 하이퍼볼릭 유사도(T0 기준)로 rerank
    #     - 쿼리/서브그래프 모두 expmap0 -> logmap0(T0) 후 같은 탄젠트에서 비교
    #     - 수치 안정화 및 NaN 안전화 포함
    #     """

    #     query_id = getattr(self, "current_query_id", None) or getattr(self, "last_query_id", None) or "q"
    #     run_id   = now_run_id()

    #     q_path = save_graph_with_ids(query_graph, query_id, run_id, "query.graphml")
    #     append_manifest_row(query_id, run_id, {
    #         "file": os.path.basename(q_path),
    #         "type": "query",
    #         "root_id": "",
    #         "nodes": query_graph.vcount(),
    #         "edges": query_graph.ecount(),
    #     })

    #     # 방어: 길이 일치
    #     top_k_root_ids = list(semantic_top_k_ids)
    #     if len(top_k_root_ids) == 0:
    #         return [], []

    #     if len(semantic_top_k_scores) != len(top_k_root_ids):
    #         # 잘못된 호출 방어: 길이를 강제로 맞춤
    #         k = len(top_k_root_ids)
    #         semantic_top_k_scores = list(semantic_top_k_scores[:k])

    #     tree_builder = TreeStructureSubgraph(self.graph)
    #     sem_scores_arr = np.array(semantic_top_k_scores)
        
    #     head_idx = np.argsort(-sem_scores_arr)[:struct_k]    
    #     # 서브그래프 추출 (상위 N개만)
    #     head_root_ids = [top_k_root_ids[i] for i in head_idx]
    #     subgraphs = tree_builder.total_subgraph(head_root_ids)

    #     for idx, (rid, sg) in enumerate(zip(head_root_ids, subgraphs)):
    #         # 서브그래프 메타(루트 등)도 그래프 속성으로 남기면 좋음
    #         sg["root_id"] = int(rid)
    #         fname = f"subgraph_{idx:03d}_root-{int(rid)}.graphml"
    #         sg_path = save_graph_with_ids(sg, query_id, run_id, fname)
    #         append_manifest_row(query_id, run_id, {
    #             "file": os.path.basename(sg_path),
    #             "type": "subgraph",
    #             "root_id": int(rid),
    #             "nodes": sg.vcount(),
    #             "edges": sg.ecount(),
    #         })
    # def rerank_with_structure(
    #     self,
    #     query_graph: ig.Graph,
    #     semantic_top_k_ids: list[int],
    #     semantic_top_k_scores: list[float],
    #     alpha: float = 0.7,
    #     embed_dim: int = 32,
    #     curvature: float = 1.0,
    #     similarity_mode: str = "cosine",
    #     struct_k : int = 30,
    #     device: str = "cpu",
    #     ) -> tuple[list[int], list[float]]:
    #     """
    #     구조 기반 하이퍼볼릭 유사도(T0 기준)로 rerank
    #     - 쿼리/서브그래프 모두 expmap0 -> logmap0(T0) 후 같은 탄젠트에서 비교
    #     - 수치 안정화 및 NaN 안전화 포함
    #     """
    #     query_id = getattr(self, "current_query_id", None) or getattr(self, "last_query_id", None) or "q"
    #     run_id   = now_run_id()

    #     q_path = save_graph_with_ids(query_graph, query_id, run_id, "query.graphml")
    #     append_manifest_row(query_id, run_id, {
    #         "file": os.path.basename(q_path),
    #         "type": "query",
    #         "root_id": "",
    #         "nodes": query_graph.vcount(),
    #         "edges": query_graph.ecount(),
    #     })

    #     # 방어: 길이 일치
    #     top_k_root_ids = list(semantic_top_k_ids)
    #     if len(top_k_root_ids) == 0:
    #         return [], []

    #     if len(semantic_top_k_scores) != len(top_k_root_ids):
    #         k = len(top_k_root_ids)
    #         semantic_top_k_scores = list(semantic_top_k_scores[:k])

    #     # --- head 선택 ---
    #     sem_scores_arr = np.array(semantic_top_k_scores, dtype=float)
    #     head_idx = np.argsort(-sem_scores_arr)[:struct_k]
    #     head_root_ids = [top_k_root_ids[i] for i in head_idx]

    #     # --- NEW: PPR 기반 2단계 서브그래프 생성 ---
    #     ppr_scores = getattr(self, "last_ppr_scores", None)  # 이미 계산/보관한 전역 PPR 점수
    #     cfg = SubgraphConfig(max_depth=2, weight_threshold=0.5,
    #                         allow_topk=200, flow_quantile=0.90, per_node_top_r=5)
    #     builder = SubgraphBuilder(self.graph, cfg)
    #     pruner  = PPRPruner(self.graph, ppr_scores, cfg)
    #     allow_ids = None
    #     if ppr_scores is not None and cfg.allow_topk > 0:
    #         allow_ids = set(np.argsort(-np.asarray(ppr_scores))[:cfg.allow_topk].tolist())


    #     subgraphs = []
    #     for rid in head_root_ids:
    #         cands  = builder.collect_candidates(root=rid, allow_ids=allow_ids)  # 1) 후보 수집
    #         pruned = pruner.prune(root=rid, candidate_edges=cands)              # 2) PPR 가지치기
    #         sg     = builder.build_subgraph(pruned)                             # 3) 서브그래프 생성
    #         subgraphs.append(sg)

    #     # 서브그래프 저장(기존 로직 유지)
    #     for idx, (rid, sg) in enumerate(zip(head_root_ids, subgraphs)):
    #         sg["root_id"] = int(rid)
    #         fname = f"subgraph_{idx:03d}_root-{int(rid)}.graphml"
    #         sg_path = save_graph_with_ids(sg, query_id, run_id, fname)
    #         append_manifest_row(query_id, run_id, {
    #             "file": os.path.basename(sg_path),
    #             "type": "subgraph",
    #             "root_id": int(rid),
    #             "nodes": sg.vcount(),
    #             "edges": sg.ecount(),
    #         })

    #     # 2) 임베더 (원점 기준 실험)
    #     embedder = HyperbolicEmbedder(embed_dim=embed_dim, curvature=curvature, device=device)

    #     # 3) 쿼리/서브그래프: expmap0 -> logmap0(T0) 바로 비교
    #     with torch.no_grad():
    #         tq = embedder.embed_and_logmap0(query_graph)  # (D,)
    #         tdocs = [embedder.embed_and_logmap0(g) for g in subgraphs]  # List[(D,)]


    #     # --- 공유 표준화 적용 ---
    #     tq_std, *tdocs_std = embedder.shared_standardize([tq] + tdocs)

    #     # 4) 구조 유사도 계산 (T0에서 코사인 등)
    #     scorer = StructralSimilarity(mode=similarity_mode)
    #     # structral_similarities = np.zeros(len(top_k_root_ids), dtype=np.float32)

    #     # compute_listwise는 q 리스트와 d 리스트 길이가 같아야 하므로 tq를 복제
    #     q_list = [tq_std for _ in range(len(tdocs_std))]
    #     struct_sims_t = scorer.compute_listwise(q_list, tdocs_std)  # torch.Tensor [k]

    #     # NaN/Inf 안전화
    #     head_struct_sims = torch.nan_to_num(struct_sims_t, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
    #     # structral_similarities[head_idx] = head_struct_sims
    #     # --- 정규화: semantic/struct 둘 다 head-20 기준 ---
    #     sem_norm_head = scorer.normalize(sem_scores_arr[head_idx])
    #     str_norm_head = scorer.normalize(head_struct_sims)

    #     # 5) 점수 정규화
    #     # sem_norm = scorer.normalize(semantic_top_k_scores)  # np.ndarray [k]
    #     # if np.all(~np.isfinite(head_struct_sims)) or np.all(head_struct_sims == 0):
    #     #     # 구조점수가 모두 비정상/0이면 구조 신호는 0으로
    #     #     str_norm = np.zeros_like(sem_norm, dtype=np.float32)
    #     # else:
    #     #     str_norm = scorer.normalize(head_struct_sims)

    #     # 6) 융합
    #     final_scores = alpha * sem_norm_head + (1.0 - alpha) * str_norm_head

    #     # --- 최종 병합: head-20 재정렬 + 나머지 그대로 ---
    #     head_pairs = sorted(zip([head_root_ids[i] for i in range(len(head_root_ids))],
    #                             final_scores), key=lambda x: x[1], reverse=True)
    #     tail_ids = [doc_id for i, doc_id in enumerate(top_k_root_ids) if i not in head_idx]

    #     reranked_doc_ids = [doc_id for doc_id, _ in head_pairs] + tail_ids
    #     reranked_scores = [score for _, score in head_pairs] + list(sem_scores_arr[[i for i in range(len(top_k_root_ids)) if i not in head_idx]])


    #     # 7) 재정렬
    #     # reranked = sorted(zip(top_k_root_ids, final_scores.tolist()), key=lambda x: x[1], reverse=True)
    #     # reranked_doc_ids = [doc_id for doc_id, _ in reranked]
    #     # reranked_scores = [score for _, score in reranked]

    #     # 디버그 (원하면 주석 해제)
    #     # print("structral_similarities:", struct_sims_t)
    #     # print("norm_semantic_top_k_scores:", sem_norm)
    #     # print("norm_structral_similarities:", str_norm)

    #     return reranked_doc_ids, reranked_scores



    # def triples_to_query_graph(self, triples: list, root_node="question"):
    #     """
    #     Converts a list of (subject, predicate, object) triples into a connected igraph.Graph,
    #     attaching root_node to the highest-degree node in each disconnected component.
    #     """
    #     if not triples:
    #         logger.warning("[TriplesToQueryGraph] Empty triples list received.")

    #         return None

    #     # 필터링
    #     valid_triples = [t for t in triples if len(t) == 3 and all(t)]
    #     if not valid_triples:
    #         logger.warning("[TriplesToQueryGraph] No valid triples after filtering.")
    #         g = ig.Graph()
    #         g.add_vertex(root_node)
    #         return g

    #     nodes = set()
    #     edges = []
    #     edge_relations = []

    #     for subj, pred, obj in valid_triples:
    #         nodes.add(subj)
    #         nodes.add(obj)
    #         edges.append((subj, obj))
    #         edge_relations.append(pred)

    #     nodes.add(root_node)

    #     g = ig.Graph()
    #     g.add_vertices(list(nodes))
    #     g.add_edges(edges)
    #     g.es["relation"] = edge_relations

    #     components = g.components()
    #     for comp in components:
    #         comp_nodes = [g.vs[idx]["name"] for idx in comp]
    #         if root_node in comp_nodes:
    #             continue
    #         degrees = [g.degree(idx) for idx in comp]
    #         rep_idx = np.argmax(degrees)
    #         rep_node = comp_nodes[rep_idx]
    #         g.add_edge(root_node, rep_node)
    #         g.es[-1]["relation"] = "root"
            

    #         logger.debug("[TriplesToQueryGraph] Final query graph summary:")
    #         logger.debug(f"- Nodes ({g.vcount()}): {[v['name'] for v in g.vs]}")
    #         logger.debug(f"- Edges ({g.ecount()}): {g.get_edgelist()}")
    #         logger.debug(f"- Relations: {g.es['relation']}")

    #     return g


    def make_query_id(self, q: str) -> str:
        # 네 코드의 compute_mdhash_id 를 재사용해도 좋음
        import hashlib
        return hashlib.md5(q.encode("utf-8")).hexdigest()[:12]

    def saver(self, g, filename: str, meta: dict):
        """
        유틸에서 넘겨주는 그래프를 <graph_debug>/<qid>/<run_id>/ 아래에 저장하고
        manifest에 메타를 한 줄씩 추가.
        """
        qid    = meta.get("query_id")    # 필수
        run_id = meta.get("run_id")      # 필수
        out_dir = os.path.join("graph_debug", qid, run_id)
        os.makedirs(out_dir, exist_ok=True)

        # save_graph_with_ids(g, qid, run_id, filename)가 이미 디렉토리를 만들면 그걸 써도 됨.
        path = os.path.join(out_dir, filename)
        g.write_graphml(path)
        print(f"[SAVE] {filename} saved to {path}")

        # manifest 갱신
        append_manifest_row(qid, run_id, {
            "file": filename,
            "type": meta.get("type", ""),
            "root_id": meta.get("root_id", ""),
            "nodes": g.vcount(),
            "edges": g.ecount(),
        })
            
        # (HippoRAG_renew 클래스 내에 추가되어야 할 함수 - 예시)
    def _save_all_debug_info(self):
        """모든 쿼리의 디버그 정보를 JSONL 파일로 저장"""
        
        if dbg is not None:
            case_dir = self._case_dir(query_id, run_id)

            # 핵심만 추려서 저장(너무 큰 건 제외)
            dbg_payload = {
                "query": query,
                "q_idx": q_idx,
                "pre_ids": dbg.get("pre_ids", None),
                "pre_scores": dbg.get("pre_scores", None),
                "reranked_doc_ids": dbg.get("reranked_doc_ids", None),
                "reranked_scores": dbg.get("reranked_scores", None),
                "rerank_kwargs": dbg.get("rerank_kwargs", {}),
                "policy": dbg.get("policy", None),
                "features": dbg.get("features", None),
                "subgraphs": dbg.get("c_subgraphs", None),  # adaptive_rerank에서 채우도록
            }
            self._dump_json(case_dir / "04_rerank_debug.json", dbg_payload)

        if not self.all_debug_info:
            logger.info("No debug info to save.")
            return

        import json
        from pathlib import Path

        # 파일명: {dataset}_full_debug_info.jsonl
        file_name = f"{self.global_config.dataset}_full_debug_info.jsonl"
        output_dir = Path(f"./outputs/{self.global_config.dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / file_name

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in self.all_debug_info:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"All debug info (DBG) saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save all debug info: {e}")
            
            
    # HippoRAG 클래스 안에 추가
    from pathlib import Path
    import json, gzip

    def _case_dir(self, query_id: str, run_id: str) -> Path:
        base = Path(self.global_config.save_dir) / "case_study" / run_id / query_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _dump_json(self, path: Path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _dump_jsonl(self, path: Path, rows: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
