from vllm import LLM, SamplingParams
from PIL import Image
import requests
import base64
from io import BytesIO
from transformers import AutoTokenizer

llm = LLM("Salesforce/blip2-opt-2.7b")

url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
# buffered = BytesIO()
# image.save(buffered, format="JPEG")
# img_bytes = buffered.getvalue()
# img_base64 = base64.b64encode(img_bytes).decode('utf-8')
# img_url = f"data:image;base64,{img_base64}"


samplingParams1 = SamplingParams(temperature=0.8, seed=0)
samplingParams2 = SamplingParams(temperature=0.8, seed=1)
samplingParams3 = SamplingParams(temperature=0.2, seed=0)

prompt = "Question: What is happening in this image? Answer:"

inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    },
}

output1 = llm.generate(inputs, samplingParams1)
output2 = llm.generate(inputs, samplingParams2)
output3 = llm.generate(inputs, samplingParams3)

print("Output with samplingParams1:", output1[0].outputs[0].text)
print("Output with samplingParams2:", output2[0].outputs[0].text)
print("Output with samplingParams3:", output3[0].outputs[0].text)

# Output with samplingParams1:  A high school kid is working on a writing assignment in a classroom with other students
# Output with samplingParams2:  A teacher is interacting with her students as they draw a map in a classroom
# Output with samplingParams3:  The teacher is teaching the students about the world map
