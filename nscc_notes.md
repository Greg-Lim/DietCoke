

qsub -I -l select=1:ngpus=1 -l walltime=3:00:00 -P personal-lims0286 -q normal
qsub -I -l select=1:ngpus=2 -l walltime=1:00:00 -P personal-lims0286 -q normal

module add git
module add python/3.12.1-gcc11
module add python/3.10.9

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/users/ntu/lims0286/scratch

nvidia-smi dmon -i 0 -s mu -d 5 -o TD > gpu.log 
fastapi dev lavis_serve.py > serve.log

CUDA_VISIBLE_DEVICES=0 python -m vllm serve "mistralai/Mistral-7B-Instruct-v0.2" --dtype float16 --host localhost --port 8001 --chat-template vllm_templates/tool_chat_template_mistral_parallel.jinja --max-model-len 16384 &