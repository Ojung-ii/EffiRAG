import time
import logging
import numpy as np
from typing import List

from tqdm import tqdm
from fastembed import SparseTextEmbedding

from .llm import _get_llm_class, BaseLLM
from .prompts.prompt_template_manager import PromptTemplateManager
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .utils.misc_utils import min_max_normalize
from .utils.config_utils import BaseConfig
from .StandardRAG import QuerySolution

logger = logging.getLogger(__name__)


class BM25_DPR:
    """
    Sparse BM25 Retriever using FastEmbed (Qdrant/bm25)

    - Retrieval: BM25 (sparse lexical matching)
    - QA / Evaluation: identical to StandardRAG
    """

    def __init__(self, global_config: BaseConfig):
        self.global_config = global_config

        # --------------------------------------------------
        # LLM (QA reader) — same as StandardRAG
        # --------------------------------------------------
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={
                "system": "system",
                "user": "user",
                "assistant": "assistant",
            }
        )

        # --------------------------------------------------
        # BM25 sparse encoder (FastEmbed)
        # --------------------------------------------------
        self.bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.docs: List[str] = []
        self.doc_sparse = []  # List[SparseEmbedding]

        self.ready_to_retrieve = False
        self.all_retrieval_time = 0.0

    # ==================================================
    # Indexing
    # ==================================================
    def index(self, docs: List[str]):
        logger.info("Indexing documents with BM25 (FastEmbed / Qdrant-bm25)")
        self.docs = docs
        self.doc_sparse = list(self.bm25.embed(docs))
        self.ready_to_retrieve = True

    # ==================================================
    # Retrieval (BM25)
    # ==================================================
    def retrieve(
        self,
        queries: List[str],
        num_to_retrieve: int = None,
        gold_docs: List[List[str]] = None,
    ):
        if not self.ready_to_retrieve:
            raise RuntimeError("BM25 index not built. Call index() first.")

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(self.global_config)

        start_time = time.time()

        # Encode queries
        query_sparse = list(self.bm25.embed(queries))

        results = []

        for q_idx, query in tqdm(
            enumerate(queries),
            total=len(queries),
            desc="BM25 Retrieving",
        ):
            q_emb = query_sparse[q_idx]
            q_map = dict(zip(q_emb.indices.tolist(), q_emb.values.tolist()))

            scores = np.zeros(len(self.doc_sparse), dtype=np.float32)

            # Sparse dot-product
            for d_idx, d_emb in enumerate(self.doc_sparse):
                s = 0.0
                for tid, val in zip(
                    d_emb.indices.tolist(),
                    d_emb.values.tolist(),
                ):
                    if tid in q_map:
                        s += q_map[tid] * val
                scores[d_idx] = s

            scores = min_max_normalize(scores)
            top_ids = np.argsort(scores)[::-1][:num_to_retrieve]

            top_docs = [self.docs[i] for i in top_ids]
            top_scores = scores[top_ids].tolist()

            results.append(
                QuerySolution(
                    question=query,
                    docs=top_docs,
                    doc_scores=top_scores,
                )
            )

        self.all_retrieval_time += time.time() - start_time
        logger.info(f"BM25 Retrieval Time: {self.all_retrieval_time:.2f}s")

        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 50, 100, 200]
            overall_result, _ = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[r.docs for r in results],
                k_list=k_list,
            )
            return results, overall_result

        return results

    # ==================================================
    # RAG-QA (same logic as StandardRAG)
    # ==================================================
    def rag_qa(self, queries, gold_docs=None, gold_answers=None):
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(self.global_config)
            qa_f1_evaluator = QAF1Score(self.global_config)

        overall_retrieval_result = None

        # 1) Retrieval
        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(
                    queries=queries, gold_docs=gold_docs
                )
            else:
                queries = self.retrieve(queries=queries)

        # 2) QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # 3) Evaluate QA + Logging (StandardRAG 스타일)
        if gold_answers is not None:
            overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[q.answer for q in queries_solutions],
                aggregation_fn=np.max
            )
            overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers,
                predicted_answers=[q.answer for q in queries_solutions],
                aggregation_fn=np.max
            )

            # QA 결과 합치고 4자리 반올림 (StandardRAG와 동일)
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_em_result.items()}

            # Retrieval 결과도 4자리 반올림 (원하면)
            if isinstance(overall_retrieval_result, dict):
                overall_retrieval_result = {k: round(float(v), 4) for k, v in overall_retrieval_result.items()}

            logger.info(f"Dataset: {self.global_config.dataset}")
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # StandardRAG처럼 gold 저장도 해주면 디버깅/분석에 유용
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return (
                queries_solutions,
                all_response_message,
                all_metadata,
                overall_retrieval_result,
                overall_qa_results
            )

        # gold_answers 없는 경우에는 출력할 QA metric이 없음
        logger.info(f"Dataset: {self.global_config.dataset}")
        logger.info("Evaluation results for retrieval: None (gold_docs not provided)")
        logger.info("Evaluation results for QA: None (gold_answers not provided)")
        return queries_solutions, all_response_message, all_metadata


    # ==================================================
    # QA (LLM Reader)
    # ==================================================
    def qa(self, queries: List[QuerySolution]):
        all_messages = []

        for q in tqdm(queries, desc="Collecting QA prompts"):
            prompt_user = ""
            for p in q.docs[: self.global_config.qa_top_k]:
                prompt_user += f"Wikipedia Title: {p}\n\n"
            prompt_user += f"Question: {q.question}\nThought: "

            dataset_name = (
                self.global_config.dataset
                if self.prompt_template_manager.is_template_name_valid(
                    f"rag_qa_{self.global_config.dataset}"
                )
                else "musique"
            )

            all_messages.append(
                self.prompt_template_manager.render(
                    name=f"rag_qa_{dataset_name}",
                    prompt_user=prompt_user,
                )
            )

        outputs = [
            self.llm_model.infer(m)
            for m in tqdm(all_messages, desc="QA Reading")
        ]

        responses, metadata, _ = zip(*outputs)

        results = []
        for i, q in enumerate(queries):
            try:
                ans = responses[i].split("Answer:")[1].strip()
            except Exception:
                ans = responses[i]

            q.answer = ans
            results.append(q)

        return results, list(responses), list(metadata)
