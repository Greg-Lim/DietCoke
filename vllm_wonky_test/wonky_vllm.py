
from openai import OpenAI
from vllm import SamplingParams
from PIL import Image
import requests
import base64
from io import BytesIO

# vllm serve "Salesforce/blip2-opt-2.7b" --dtype float16 --host localhost --port 8000 --chat-template vllm_templates/template_blip2.jinja --gpu-memory-utilization 0.4

url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

sampling_params = SamplingParams(temperature=0.6)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="idk"
)

prompt = "Question: What is happening in this image? Answer:"

buffered = BytesIO()
image.save(buffered, format="JPEG")
img_bytes = buffered.getvalue()
img_base64 = base64.b64encode(img_bytes).decode('utf-8')
img_url = f"data:image;base64,{img_base64}"
messages = [{
        "role": "assistant",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": img_url
                }
            }
        ]
    }]

samplingParams1 = SamplingParams(temperature=0.8, seed=0)
samplingParams2 = SamplingParams(temperature=0.8, seed=1)
samplingParams3 = SamplingParams(temperature=0.2, seed=0)

output1 = client.chat.completions.create(model="Salesforce/blip2-opt-2.7b", messages=messages, temperature=samplingParams1.temperature)
output2 = client.chat.completions.create(model="Salesforce/blip2-opt-2.7b", messages=messages, temperature=samplingParams2.temperature)
output3 = client.chat.completions.create(model="Salesforce/blip2-opt-2.7b", messages=messages, temperature=samplingParams3.temperature)


print("Output with samplingParams1:", output1.choices[0].message.content)
print("Output with samplingParams2:", output2.choices[0].message.content)
print("Output with samplingParams3:", output3.choices[0].message.content)