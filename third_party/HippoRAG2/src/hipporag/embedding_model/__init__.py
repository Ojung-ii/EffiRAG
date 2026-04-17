from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .GTR import GTREmbeddingModel
from .GTEQwen2 import GTEQwen2EmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    name = embedding_model_name.lower()

    if "gritlm" in name:
        return GritLMEmbeddingModel
    elif "nv-embed-v2" in name:
        return NVEmbedV2EmbeddingModel
    elif "gte-qwen2" in name or "alibaba-nlp/gte-qwen2" in name:
        return GTEQwen2EmbeddingModel
    elif "contriever" in name:
        return ContrieverModel
    elif "gtr" in name:
        return GTREmbeddingModel
    elif "text-embedding" in name:
        return OpenAIEmbeddingModel

    assert False, f"Unknown embedding model name: {embedding_model_name}"
