import base64
from io import BytesIO
import time
from vllm import SamplingParams
from PIL import Image
import requests
import concurrent.futures as futures
from openai import OpenAI

question = "This image shows"
question = "What is happening in this image?" # this prompt is slightly better

def generate_multiple_captions_vllm(client, model, image, num_captions, sampling_params=None):
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

def generate_multiple_captions_vllm_multi(client, model, image, num_captions, sampling_params=None):
    '''
    This method is about 10% slower but supports unlimited size of num_captions without breaking vllm
    '''
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.9)
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    img_url = f"data:image;base64,{img_base64}"

    def multi_threaded_chat_completion(seed = 0):
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
        return client.chat.completions.create(model=model, messages=messages, temperature=sampling_params.temperature, max_completion_tokens=100, seed=seed)
    
    with futures.ThreadPoolExecutor() as executor:
        outputs = list(executor.map(multi_threaded_chat_completion, range(num_captions)))

    return [output.choices[0].message.content+"." for output in outputs]
    
if __name__ == "__main__":
    num_captions = 30

    url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    sampling_params = SamplingParams(temperature=0.6)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="idk"
    )
    num_captions = 30
    tick = time.time()
    captions = generate_multiple_captions_vllm(client, "Salesforce/blip2-opt-2.7b", image, num_captions, sampling_params=sampling_params)
    time_vllm = time.time() - tick
    print("Generated Caption with blip2 vllm:", captions)
    print("Unique captions:", len(set(captions)))
    print("Time taken generate_multiple_captions_vllm:", time_vllm)

    num_captions = 200
    tick = time.time()
    captions = generate_multiple_captions_vllm_multi(client, "Salesforce/blip2-opt-2.7b", image, num_captions, sampling_params=sampling_params)
    time_vllm = time.time() - tick
    print("Generated Caption with blip2 vllm:", captions)
    print("Unique captions:", len(set(captions)))
    print("Time taken generate_multiple_captions_vllm_multi:", time_vllm)
