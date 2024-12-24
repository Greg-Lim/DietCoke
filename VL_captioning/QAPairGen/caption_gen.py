import base64
from io import BytesIO
import time
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from vllm import SamplingParams
import torch
from PIL import Image
import requests

from openai import OpenAI


def generate_multiple_captions(engine, num_captions, image, **kwargs):
    if engine == "huggingface":
        return _generate_multiple_captions_hf(num_captions=num_captions, image=image, **kwargs)
    elif engine == "vllm":
        return _generate_multiple_captions_vllm(num_captions=num_captions, image=image, **kwargs)

question = "This image shows"

def _generate_multiple_captions_hf(processor, model, image, num_captions, **kwargs):
    captions = []
    # for _ in range(num_captions):
    inputs = processor([image]*num_captions, [question]*num_captions, return_tensors="pt").to(model.device)
    generation_args = {
        "temperature": 0.1,
        "repetition_penalty": 1.2,
        "min_new_tokens": 16,
        "max_new_tokens": 64,
        "do_sample": True,
        "use_cache": False,
    }
    generated_ids = model.generate(**inputs, **generation_args)
    captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions

def _generate_multiple_captions_vllm(client, model, image, num_captions, sampling_params=None):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.6)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    img_url = f"data:image;base64,{img_base64}"
    messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    }
                }
            ]
        }]

    outputs = client.chat.completions.create(model=model, messages=messages, temperature=sampling_params.temperature, n=num_captions, max_completion_tokens=100)
    print(outputs)
    return [output.message.content+"." for output in outputs.choices]

    
if __name__ == "__main__":
    num_captions = 30

    url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    if 1 and "skip hf":
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tick = time.time()
        captions = generate_multiple_captions(engine="huggingface", processor=processor, model=model, image=image, num_captions=num_captions)
        time_hf = time.time() - tick
        print("Generated Caption with blip2 hf:", captions)
        print("Time taken hf:", time_hf)
    # 2 Cpations: 5.6 seconds
    # 30 Captions: 29.1 seconds

    if 0 and "skip vllm":
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.2)
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="idk"
        )

        tick = time.time()
        captions = generate_multiple_captions(engine="vllm", model= "Salesforce/blip2-opt-2.7b", client=client, image=image, num_captions=num_captions, sampling_params=sampling_params)
        time_vllm = time.time() - tick
        print("Generated Caption with blip2 vllm:", captions)
        print("Time taken vllm:", time_vllm)
    # 2 captions: 0.21 seconds
    # 30 captions: 0.76 seconds

    
