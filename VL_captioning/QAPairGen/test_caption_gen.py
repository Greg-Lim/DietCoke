from openai import OpenAI
import pytest
from PIL import Image
import requests
from caption_gen import generate_multiple_captions_vllm
from vllm import SamplingParams

@pytest.fixture(scope="module")
def client():
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="idk"
    )
    return client

@pytest.fixture(scope="module")
def image():
    url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
    return Image.open(requests.get(url, stream=True).raw).convert('RGB')

# pytest -v VL_captioning/QAPairGen/test_caption_gen.py -k test_generate_multiple_captions_vllm -s
def test_generate_multiple_captions_vllm(client, image):
    num_captions = 30
    sampling_params = SamplingParams(temperature=0.6)
    captions = generate_multiple_captions_vllm(client, "Salesforce/blip2-opt-2.7b", image, num_captions, sampling_params)
    print("vllm Captions:", captions)
    print("Total Unique:", len(set(captions)))
    assert isinstance(captions, list)
    assert len(captions) == num_captions
    unique_captions = len(set(captions))
    if unique_captions < num_captions / 2:
        print("Warning: Less than half of the captions are unique.")
    for caption in captions:
        assert isinstance(caption, str)
        assert len(caption) > 0