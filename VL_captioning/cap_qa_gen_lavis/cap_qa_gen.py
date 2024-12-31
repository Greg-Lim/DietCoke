import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.models import load_model_and_preprocess
from lavis.common.gradcam import getAttMap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_caption_qa(lavis_model, vis_processors, txt_processors, raw_image, question, config):
    assert "num_caps_per_img" in config, "Config must contain 'num_caps_per_img' key"
    assert "num_question_per_img" in config, "Config must contain 'num_question_per_img' key"

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    samples = {"image": image, "text_input": [question]}

    samples = lavis_model.forward_itm(samples=samples)
    samples = lavis_model.forward_cap(
        samples=samples, 
        num_captions=config.get("num_caps_per_img", None),
        top_k=config.get("top_k", None),
        num_patches=20)
    samples = lavis_model.forward_qa_generation(samples)

    return samples["captions"], samples["questions"], samples["answers"]



if __name__ == '__main__':
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/pnp-vqa/demo.png'
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    raw_image = Image.open("./VL_captioning/cap_qa_gen_lavis/demo.png").convert("RGB")
    question = "What item s are spinning which can be used to control electric?"
    print(question)

    lavis_model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

    config = {
        "num_caps_per_img": 5,
        "num_question_per_img": 5
    }

    captions, questions, answers = get_caption_qa(lavis_model, vis_processors, txt_processors, raw_image, question, config)
    print(captions)
    print(questions, len(questions))
    print(answers)
    


