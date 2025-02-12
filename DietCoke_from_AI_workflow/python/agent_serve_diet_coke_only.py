
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
from DietCokeComponents.dietcoke_only import diet_coke_e2e

# python benchmarks/DietCoke/python/agent_serve.py 
# {"question": "What is behind the people?", "image_path": "benchmarks/DietCoke/python/demo2.jpg"}


llm_url = "http://localhost:8090/v1"

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

    print("Starting Diet Coke Agent")
    config = init()

    input_data = sys.stdin.read()
    request = json.loads(input_data)

    # request = {'image_id': 318245, 'question': 'The fabric on that couch was very popular in the eighties what was it called?', 'question_id': 3182455, 'request_id': 848}
    print("request", request)

    llm_client = OpenAI(
        api_key="idk",
        base_url=llm_url,
    )


    question = request["question"]

    with open(config["aokvqa_caption_question_files_path"]+"/aokvqa_val_caption.json", "r") as f:
        caption_list = json.load(f)

    with open(config["aokvqa_caption_question_files_path"]+"/aokvqa_val_question.json", "r") as f:
        question_list = json.load(f)

    with open(config["aokvqa_caption_question_files_path"]+"/aokvqa_val_ans_to_cap_dict.json", "r") as f:
        ans_dict_queid_list = json.load(f)

    def find_in(list, key, target):
        for entry in list:
            if entry[key] == target:
                return entry

    captions = find_in(caption_list, "question_id", request["question_id"])["caption"]
    syn_qa = find_in(question_list, "question_id", request["question_id"])
    syn_question = syn_qa["question"]
    syn_answer = syn_qa["answer"]
    ans_dict_queid = find_in(ans_dict_queid_list, "question_id", request["question_id"])["ans_to_cap_dict"]



    tik = time.time()
    ans = diet_coke_e2e(llm_client, question, ans_dict_queid, syn_question, syn_answer, captions, config)
    tok = time.time()

    print("Time taken: ", tok-tik)

    print(ans)