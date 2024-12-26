




# NSCC (A100 40GB)
``` bash
vllm serve "mistralai/Mistral-7B-v0.1" --dtype float16 --host localhost --port 8001 --download-dir /scratch/users/ntu/lims0286 --gpu-memory-utilization 0.45 --max-model-len 4096 &
vllm serve "mistralai/Mistral-7B-v0.1" --dtype float16 --host localhost --port 8001 --download-dir /scratch/users/ntu/lims0286 --gpu-memory-utilization 0.6 &
^ not work


vllm serve "Salesforce/blip2-opt-2.7b" --dtype half --host localhost --port 8000 --download-dir /scratch/users/ntu/lims0286 --gpu-memory-utilization 0.35 --max-model-len 512 &
vllm serve "Salesforce/blip2-opt-2.7b" --dtype half --host localhost --port 8000 --download-dir /scratch/users/ntu/lims0286 --gpu-memory-utilization 0.8 --max-model-len 512 &

```

gpu-memory-utilization is of remaining memory.

# V100 32GB
``` bash
vllm serve "mistralai/Mistral-7B-v0.1" --dtype float16 --host localhost --port 8001 --chat-template vllm_templates/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.6 --max-model-len 16384
vllm serve "mistralai/Mistral-7B-Instruct-v0.2" --dtype float16 --host localhost --port 8001 --chat-template vllm_templates/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.6 --max-model-len 16384
vllm serve "Salesforce/blip2-opt-2.7b" --dtype float16 --host localhost --port 8000 --chat-template vllm_templates/template_blip2.jinja --gpu-memory-utilization 0.4
vllm serve "Salesforce/blip2-opt-2.7b" --dtype float16 --host localhost --port 8000 --chat-template vllm_templates/template_blip2.jinja --gpu-memory-utilization 0.4 --enable-prefix-caching
```

Data set can be gotten from https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA