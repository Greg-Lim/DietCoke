import torch
from lavis.models import load_model_and_preprocess
import requests

import torch
from PIL import Image

from caption_bench.dataset import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

def get_lavis_caption(image, question) -> list[str]:    
    lavis_image = vis_processors["eval"](image).unsqueeze(0).to(device)
    lavis_question = txt_processors["eval"](question)
    lavis_samples = {"image": lavis_image, "text_input": [lavis_question]}
    lavis_samples = model.forward_itm(samples=lavis_samples)
    lavis_samples = model.forward_cap(samples=lavis_samples, num_captions=50, num_patches=20)

    return lavis_samples['captions'][0]

def get_lavis_QA(captions) -> dict:
    lavis_samples = {"captions": captions}
    lavis_samples = model.forward_qa_generation(lavis_samples)
    lavis_samples.pop("captions", None)

    return lavis_samples

if __name__ == "__main__":
    data = get_data(1)

    for d in data:
        image = d["image"]
        question = d["question"]
        captions = get_lavis_caption(image, question)
        print(captions)
        t = get_lavis_QA(captions)
        print(t)
        