vllm serve \
     Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size 1 \
    --max_model_len 16384 \
    --port 8011
