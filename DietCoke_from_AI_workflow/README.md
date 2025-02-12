# DietCoke

## Prerequisites

- Python 3.10 or later

## Installation

1. Clone the AI benchmark repository:
```bash
git clone https://github.com/vhive-serverless/AI-workflow-benchmark.git
cd benchmark/DietCoke
```

2. Initialize DietCoke environment:
```bash
virtualenv --python="/usr/bin/python3.10" venv
source venv/bin/activate
pip install -r requirement.txt
```

3. Clone the LLMLoadgen

```
git clone https://github.com/vhive-serverless/LLMLoadgen.git
git checkout workflow
```

## Usage

1. Launch vllm as the backend 

```
vllm serve "mistralai/Mistral-7B-Instruct-v0.2" --dtype float16 --host localhost --port 8090 --chat-template benchmarks/DietCoke/python/vllm_templates/tool_chat_template_mistral_parallel.jinja
```

2. Launch DietCoke relay

```
go run relay.go
```

3. Launch LLMLoadgen

```
./LLMLoadgen -pattern {constant/burst} -sim_dir ./plots -dataset okvqa -dst benchmark -workflow dietcoke -ip localhost -port {port} -max_drift {drift}
./LLMLoadgen -pattern constant-1 -sim_dir ./plots -dataset aokvqa_caption_question -dst benchmark -workflow dietcoke -address dietcoke -ip localhost -port 8080 -max_drift 100
```