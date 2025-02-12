
import os
import yaml
import argparse
import sys
import json
from openai import OpenAI
import requests
from io import BytesIO
from PIL import Image
import time

# from agent_serve_offline_lavis_serve import diet_coke_e2e_lavis_serve
from DietCokeComponents.dietcoke_full import diet_coke_e2e_lavis_serve

# python benchmarks/DietCoke/python/agent_serve.py 
# {"question": "What is behind the people?", "image_path": "benchmarks/DietCoke/python/demo2.jpg"}


llm_url = "http://localhost:8091/v1"
lavis_url = "http://localhost:8090/"

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
CONFIG = os.path.join(script_dir, "config.yaml")

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def init():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = load_config(CONFIG)
    return config
    
if __name__ == "__main__":
    config = init()

    input_data = sys.stdin.read()
    request = json.loads(input_data)

    # input_data = input()
    # request = json.loads(input_data)

    # request = {"question": "What are the people doing?", "image_path": "benchmarks/DietCoke/python/demo2.jpg"}
    # request = {'image_id': 488011, 'question_id': 4880115, 'request_id': 2620}
    # with open(config["question_path"], "r") as f:
    #     question = json.load(f)["questions"][]

    # request = {'image_id': 318245, 'question': 'The fabric on that couch was very popular in the eighties what was it called?', 'question_id': 3182455, 'request_id': 848}

    print(request)

    # third_party/LLMLoadgen/datasets/DietCoke/ok_vqa_data/train2014/COCO_train2014_000000000009.jpg
    # third_party/LLMLoadgen/datasets/DietCoke/ok_vqa_data/train2014/COCO_train2014_000000336569.jpg
    
    imageid = str(request["image_id"]).zfill(12)
    # imagepath = "../"+config["image_folder_path"] +"/COCO_train2014_"+ imageid + ".jpg"
    # imagepath = config["image_folder_path"] +"/COCO_train2014_"+ imageid + ".jpg"
    image_folder= "/home/users/ntu/lims0286/scratch/AI-workflow-benchmark/third_party/LLMLoadgen/datasets/DietCoke/ok_vqa_data/train2014"
    imagepath = os.path.join(image_folder, "COCO_train2014_"+ imageid + ".jpg")
    print(imagepath)

    print("image loaded")

    llm_client = OpenAI(
        api_key="idk",
        base_url=llm_url,
    )

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

    # request["image_id"]
    image = Image.open(imagepath)
    image = image.convert("RGB")
    question = request["question"]

    tik = time.time()
    ans = diet_coke_e2e_lavis_serve(llm_client, get_lavis_smaples, image, question, config)
    tok = time.time()

    print("Time taken: ", tok-tik)

    print(ans)