import torch
from lavis.models import load_model_and_preprocess
import requests

import torch
from PIL import Image
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
question = "What is behind the people in the image?"

model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)

lavis_image = vis_processors["eval"](image).unsqueeze(0).to(device)
lavis_question = txt_processors["eval"](question)
lavis_samples = {"image": lavis_image, "text_input": [lavis_question]}
lavis_samples = model.forward_itm(samples=lavis_samples)
lavis_samples = model.forward_cap(samples=lavis_samples, num_captions=50, num_patches=20)
lavis_samples = model.forward_qa_generation(lavis_samples)

print(lavis_samples['captions'][0])
print(lavis_samples['questions'])
print(lavis_samples['answers'])
print(lavis_samples['ans_to_cap_dict'])
