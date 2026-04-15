import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, L, H], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts


class GTREmbeddingModel(BaseEmbeddingModel):
    """
    GTR encoder wrapper for sentence-transformers/gtr-t5-* and google/gtr-t5-*.
    Implements batch_encode(texts, instruction=..., norm=True/False).
    """

    def __init__(self, global_config=None, embedding_model_name=None):
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        else:
            self.embedding_model_name = self.global_config.embedding_model_name

        # Build embedding_config compatible with your existing pattern
        self.embedding_config = EmbeddingConfig.from_dict({
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": False,  # GTR는 보통 필요 없음
                "torch_dtype": getattr(torch, str(self.global_config.embedding_model_dtype).replace("torch.", ""), torch.float16)
                if self.global_config.embedding_model_dtype is not None else None,
            },
            "tokenizer_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "use_fast": True
            }
        })

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load
        model_params = dict(self.embedding_config.model_init_params)
        # Bert류에서 문제가 됐던 device_map은 여기서도 일단 제거(안전)
        model_params.pop("device_map", None)

        self.tokenizer = AutoTokenizer.from_pretrained(**self.embedding_config.tokenizer_init_params)
        self.model = T5EncoderModel.from_pretrained(**model_params)
        self.model.to(self.device)
        self.model.eval()

        # infer embedding dim
        with torch.no_grad():
            dummy = self.tokenizer("hello", return_tensors="pt")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.model(**dummy)
            pooled = mean_pooling(out.last_hidden_state, dummy["attention_mask"])
            self.embedding_dim = pooled.shape[-1]

        # optional disk cache (repo 패턴에 맞춰 사용)
        cache_path = getattr(self.global_config, "embedding_cache_path", None)
        if cache_path:
            self._encode_impl = make_cache_embed(self._encode_impl, cache_path, self.device)

    @torch.no_grad()
    def _encode_impl(self, prompts: list, instruction: str = "", max_length: int = 256) -> torch.Tensor:
        # instruction을 prefix로 붙이는 방식(기존 get_query_instruction과 호환)
        if instruction:
            texts = [instruction + "\n" + p for p in prompts]
        else:
            texts = prompts

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)

        emb = mean_pooling(out.last_hidden_state, batch["attention_mask"])  # [B, H]
        return emb

    @torch.no_grad()
    def batch_encode(self, texts, instruction: str = "", norm: bool = True, max_length: int = None, batch_size: int = None):
        if isinstance(texts, str):
            texts = [texts]

        if max_length is None:
            max_length = getattr(self.global_config, "embedding_max_seq_len", 256)

        if batch_size is None:
            batch_size = getattr(self.global_config, "embedding_batch_size", 32)

        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            embs = self._encode_impl(prompts=chunk, instruction=instruction, max_length=max_length)
            if norm:
                embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs)

        return torch.cat(all_embs, dim=0).detach().cpu().numpy()
