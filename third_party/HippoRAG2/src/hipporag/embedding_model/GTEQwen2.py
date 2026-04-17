import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .base import BaseEmbeddingModel, EmbeddingConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class GTEQwen2EmbeddingModel(BaseEmbeddingModel):
    """
    Wrapper for Alibaba-NLP/gte-Qwen2-7B-instruct (text embedding).
    Implements batch_encode(texts, instruction=..., norm=True/False).
    """

    def __init__(self, global_config=None, embedding_model_name=None):
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
        else:
            self.embedding_model_name = self.global_config.embedding_model_name

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # config (HippoRAG2 패턴에 맞춤)
        self.embedding_config = EmbeddingConfig.from_dict({
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,  # GTE는 remote code 필요한 경우가 있어 True 권장
                "torch_dtype": self.global_config.embedding_model_dtype,
            },
            "tokenizer_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "use_fast": True
            }
        })

        model_params = dict(self.embedding_config.model_init_params)
        model_params.pop("device_map", None)  # 안전

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=False,
            use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=False,
            torch_dtype=self.global_config.embedding_model_dtype,
            attn_implementation="eager",
        )

        self.model.to(self.device)
        self.model.eval()

        # infer embedding dim
        with torch.no_grad():
            dummy = self.tokenizer("hello", return_tensors="pt")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.model(**dummy)
            pooled = mean_pooling(out.last_hidden_state, dummy["attention_mask"])
            self.embedding_dim = pooled.shape[-1]

    @torch.no_grad()
    def batch_encode(self,
                     texts,
                     instruction: str = "",
                     norm: bool = True,
                     max_length: int = None,
                     batch_size: int = None):

        if isinstance(texts, str):
            texts = [texts]

        if max_length is None:
            max_length = getattr(self.global_config, "embedding_max_seq_len", 256)

        if batch_size is None:
            batch_size = getattr(self.global_config, "embedding_batch_size", 8)  # 7B라 보수적으로

        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]

            if instruction:
                chunk = [instruction + "\n" + t for t in chunk]

            batch = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            out = self.model(**batch)
            embs = mean_pooling(out.last_hidden_state, batch["attention_mask"])

            if norm:
                embs = F.normalize(embs, p=2, dim=-1)

            all_embs.append(embs.detach().cpu())

        return torch.cat(all_embs, dim=0).numpy()
