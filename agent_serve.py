from VL_captioning.agent_serve_offline_lavis_serve import diet_coke_e2e_lavis_serve, update
import fastapi
from fastapi import UploadFile, File, Form
from PIL import Image
from io import BytesIO
import requests
from openai import OpenAI
from collections import namedtuple
import os
import ruamel.yaml as yaml


app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


Args = namedtuple('Args', [
    'config', 'caption_file', 'question_file', 'question_ppl_file', 'ans_dict_file', 'question_type', 
    'output_dir', 'resume', 'evaluate_direct', 'evaluate', 'vqa_eval', 'device', 'seed', 'split_seed', 
    'num_sample', 'ensemble', 'random_question', 'test_server', 'dist_selection', 'select_cap', 
    'dataset', 'result_tag', 'batch_size_test', 'num_caps_per_img', 'num_question_per_img', 'model', 'llm_model', 'vlm_model'
])

args = Args(
    config='./configs/AOKVQA_caption.yaml',
    caption_file='../caption_question_files/aokvqa_val_caption.json',
    question_file='../caption_question_files/aokvqa_val_question.json',
    question_ppl_file=None,
    ans_dict_file='../caption_question_files/aokvqa_val_ans_to_cap_dict.json',
    question_type='g_q',
    output_dir='./output_knowledge_aok/mistral_inst/',
    resume=False,
    evaluate_direct=False,
    evaluate=False,
    vqa_eval=False,
    device='cuda',
    seed=42,
    split_seed=0,
    num_sample=16,
    ensemble=1,
    random_question=False,
    test_server=False,
    model='mistralai/Mistral-7B-Instruct-v0.2',
    llm_model='mistralai/Mistral-7B-Instruct-v0.2',
    vlm_model='Salesforce/blip2-opt-2.7b',
    dist_selection='hugging',
    select_cap=False,
    dataset='aokvqa',
    result_tag='',
    batch_size_test=1,
    num_caps_per_img=60,
    num_question_per_img=30
)

llm_client = OpenAI(
    api_key="idk",
    base_url="http://localhost:8001/v1",
)

lavis_url = "http://localhost:8000"
def get_lavis_smaples(image, question):
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "question": question,
    }

    response = requests.post(f"{lavis_url}/generate_caption_qa", files=files, data=data)
    response.raise_for_status()
    return response.json()

config_path = "./VL_captioning/configs/AOKVQA_caption.yaml"
print("Current directory:", os.getcwd())

yaml_loader = yaml.YAML(typ='rt')
with open(config_path, 'r') as file:
    config = yaml_loader.load(file)
config = update(config, args)

@app.post("/diet_coke")
async def generate_caption_qa(file: UploadFile = File(...), question: str = Form(...)):
    diet_coke_e2e_lavis_serve()