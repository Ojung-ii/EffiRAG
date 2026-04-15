import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
# from .vllm_client import VLLMGPT  # (직접 구현 필요

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
 #   if hasattr(config, "llm_mode") and config.llm_mode == "offline":
  #      return VLLMGPT(config.llm_base_url, config.llm_name)
    
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'
    return CacheOpenAI.from_experiment_config(config)


