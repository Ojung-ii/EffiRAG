vllm serve \
     Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.25 \
    --tensor-parallel-size 1 \
    --max_model_len 4096 \
    --port 8011
