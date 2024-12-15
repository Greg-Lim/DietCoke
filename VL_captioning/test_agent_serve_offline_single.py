from agent_serve_offline_single import diet_coke_e2e
from PIL import Image
import json
from openai import OpenAI



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

            qa.append((image, question, answers))
        return qa

image_path = "/home/greg/FYP/dataset/vqa_data/train2014"
vqa_annotations_path = "/home/greg/FYP/dataset/vqa_data/v2_mscoco_train2014_annotations.json"
vqa_question_path = "/home/greg/FYP/dataset/vqa_data/v2_OpenEnded_mscoco_train2014_questions.json"

# pytest -s test_agent_serve_offline_single.py
def test_diet_coke_e2e():
    
    qa = Dataset(image_path, vqa_annotations_path, vqa_question_path).get_image_and_qa(10)

    '''
    vllm serve Salesforce/blip2-opt-2.7b --dtype float16 --host localhost --port 8000 --chat-template ./vllm/examples/template_blip2.jinja --gpu-memory-utilization 0.3
    vllm serve "mistralai/Mistral-7B-v0.1" --dtype float16 --host localhost --port 8002 --chat-template ./vllm/examples/tool_chat_template_mistral_parallel.jinja --gpu-memory-utilization 0.7
    '''
 
    vlm_client = OpenAI(
        api_key="idk",
        base_url="http://localhost:8000",
    )

    llm_client = OpenAI(
        api_key="idk",
        base_url="http://localhost:8002",
    )

    for image, question, answers in qa:
        diet_coke_e2e(image, question, answers)