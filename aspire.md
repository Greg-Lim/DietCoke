



```bash
module add python/3.10.9
python -m venv vllm_lavis
source vllm_lavis/bin/activate

pip install -r lavis_requirements.txt

pip install vllm

vllm serve "mistralai/Mistral-7B-Instruct-v0.2" --dtype float16 --host localhost --port 8001 --chat-template vllm_templates/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.6 --max-model-len 16384 --download-dir /scratch/users/ntu/lims0286 &

```

export CUDA_VISIBLE_DEVICES=0
