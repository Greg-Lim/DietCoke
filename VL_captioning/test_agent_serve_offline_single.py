import time
from PIL import Image
import json
from openai import OpenAI
import os
from collections import namedtuple

import ruamel.yaml as yaml
from datasets import load_dataset

import os

if "NCSS":
    # Set the scratch folder path
    scratch_folder = "/home/users/ntu/lims0286/scratch"

    # Update environment variables
    os.environ["NLTK_DATA"] = os.path.join(scratch_folder, "nltk_data")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(scratch_folder, "transformers_cache")

    # Ensure the directories exist
    os.makedirs(os.environ["NLTK_DATA"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train+test+validation")

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

class Dataset:
    def __init__(self, image_path:str, vqa_annotations_path:str, vqa_question_path:str):
        self.image_path = image_path
        self.vqa_annotations_path = vqa_annotations_path
        self.vqa_question_path = vqa_question_path
        pass

    def get_image_and_qa(self, idxs: list[int] | int) -> list[tuple[Image.Image, str, list[str]]]:
        if type(idxs) == int:
            idxs = list(range(idxs))
        qa = []
        with open(self.vqa_annotations_path, 'r') as file:
            vqa_annotations = json.load(file)

        with open(self.vqa_question_path, 'r') as file:
            vqa_questions = json.load(file) 

        for i in idxs:
            image_id = vqa_annotations['annotations'][i]['image_id']
            question = vqa_questions['questions'][i]['question']
            answers = vqa_annotations['annotations'][i]['answers']
            # Construct the image file path
            image_path = f'{self.image_path}/COCO_train2014_{image_id:012d}.jpg'
            image = Image.open(image_path)

            qa.append((image, question, answers, image_id))
        return qa

image_path = "/home/greg/FYP/dataset/vqa_data/train2014"
vqa_annotations_path = "/home/greg/FYP/dataset/vqa_data/v2_mscoco_train2014_annotations.json"
vqa_question_path = "/home/greg/FYP/dataset/vqa_data/v2_OpenEnded_mscoco_train2014_questions.json"

# pytest -s VL_captioning/test_agent_serve_offline_single.py
def test_diet_coke_e2e():
    from agent_serve_offline_custom import diet_coke_e2e, update

    # qa = Dataset(image_path, vqa_annotations_path, vqa_question_path).get_image_and_qa(50)[30:35]

    qa = ds


    '''
    vllm serve Salesforce/blip2-opt-2.7b --dtype half --host localhost --port 8000 --chat-template ./vllm/examples/template_blip2.jinja --gpu-memory-utilization 0.3
    vllm serve "mistralai/Mistral-7B-v0.1" --dtype float32 --host localhost --port 8001--chat-template ./vllm/examples/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.7
    '''

    '''
    vllm serve Salesforce/blip2-opt-2.7b --dtype auto --host localhost --port 8000 --chat-template ./vllm/examples/template_blip2.jinja --gpu-memory-utilization 0.3
    vllm serve "mistralai/Mistral-7B-v0.1" --dtype auto --host localhost --port 8001 --chat-template ./vllm/examples/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.7
    '''
 
    vlm_client = OpenAI(
        api_key="idk",
        base_url="http://localhost:8000/v1",
    )

    llm_client = OpenAI(
        api_key="idk",
        base_url="http://localhost:8001/v1",
    )

    config_path = "./VL_captioning/configs/AOKVQA_caption.yaml"
    print("Current directory:", os.getcwd())

    yaml_loader = yaml.YAML(typ='rt')
    with open(config_path, 'r') as file:
        config = yaml_loader.load(file)
    config = update(config, args)

    all_dict = {}
    all_dict["pred_ans"] = []
    all_dict["actual_ans"] = []
    all_dict["question_id"] = []
    all_dict["time_taken"] = []
    for entry in ds:
        # features: ['image', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'],
        image = entry['image']
        question_id = entry['question_id']
        question = entry['question']
        choices = entry['choices']
        correct_choice_idx = entry['correct_choice_idx']
        direct_answers = entry['direct_answers']
        difficult_direct_answer = entry['difficult_direct_answer']
        rationales = entry['rationales']
        answers = entry['direct_answers']
        
        tick = time.time()
        dc_ans = diet_coke_e2e(llm_client, vlm_client, image, question, config)
        all_dict["pred_ans"].append(dc_ans)
        all_dict["actual_ans"].append(answers)
        all_dict["question_id"].append(question_id)
        all_dict["time_taken"].append(time.time()-tick)

        if True:
            print("Question:", question)
            print("Pred_ans:", dc_ans)
            print("Actual_ans:", answers)
            print("question_id:", question_id)
            print("Time taken:", time.time()-tick)
            print()
            input("Press enter to continue")

    avg_time = sum(all_dict["time_taken"]) / len(all_dict["time_taken"])
    for i in range(len(all_dict["pred_ans"])):
        print("Pred_ans:", all_dict["pred_ans"][i], "Actual_ans:", all_dict["direct_answers"][i], "question_id:", all_dict["question_id"][i],  "Time taken:", all_dict["time_taken"][i])
    print("Average time taken:", avg_time, f"over {len(all_dict['time_taken'])} images")


def test_diet_coke_e2e_lavis():
    from agent_serve_offline_lavis import diet_coke_e2e_lavis, update
    llm_client = OpenAI(
        api_key="idk",
        base_url="http://localhost:8001/v1",
    )

    import torch
    from lavis.models import load_model_and_preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    

    config_path = "./VL_captioning/configs/AOKVQA_caption.yaml"
    print("Current directory:", os.getcwd())

    yaml_loader = yaml.YAML(typ='rt')
    with open(config_path, 'r') as file:
        config = yaml_loader.load(file)
    config = update(config, args)

    all_dict = {}
    all_dict["pred_ans"] = []
    all_dict["actual_ans"] = []
    all_dict["question_id"] = []
    all_dict["time_taken"] = []
    for entry in ds:
        # features: ['image', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'],
        image = entry['image']
        question_id = entry['question_id']
        question = entry['question']
        choices = entry['choices']
        correct_choice_idx = entry['correct_choice_idx']
        direct_answers = entry['direct_answers']
        difficult_direct_answer = entry['difficult_direct_answer']
        rationales = entry['rationales']
        answers = entry['direct_answers']
        
        tick = time.time()
        dc_ans = diet_coke_e2e_lavis(llm_client, model, vis_processors, txt_processors, image, question, config)
        all_dict["pred_ans"].append(dc_ans)
        all_dict["actual_ans"].append(answers)
        all_dict["question_id"].append(question_id)
        all_dict["time_taken"].append(time.time()-tick)

        if True:
            print("Question:", question)
            print("Pred_ans:", dc_ans)
            print("Actual_ans:", answers)
            print("question_id:", question_id)
            print("Time taken:", time.time()-tick)
            print()
            input("Press enter to continue")

    avg_time = sum(all_dict["time_taken"]) / len(all_dict["time_taken"])
    for i in range(len(all_dict["pred_ans"])):
        print("Pred_ans:", all_dict["pred_ans"][i], "Actual_ans:", all_dict["direct_answers"][i], "question_id:", all_dict["question_id"][i],  "Time taken:", all_dict["time_taken"][i])
    print("Average time taken:", avg_time, f"over {len(all_dict['time_taken'])} images")


if __name__ == "__main__":
    # test_diet_coke_e2e()
    test_diet_coke_e2e_lavis()