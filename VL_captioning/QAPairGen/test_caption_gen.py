from openai import OpenAI
import pytest
from unittest.mock import MagicMock
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import requests
from caption_gen import generate_multiple_captions
from vllm import LLM, SamplingParams


@pytest.fixture(scope="module")
def hf_processor():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor

@pytest.fixture(scope="module")
def hf_model():
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

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

# pytest -v QAPairGen/test_caption_gen.py -k test_hf_generate_multiple_captions -s
def test_hf_generate_multiple_captions(hf_model, hf_processor, image):
    num_captions = 10
    captions = generate_multiple_captions("huggingface", num_captions, image, model=hf_model, processor=hf_processor)
    print("hf Captions:", captions)
    print("Total Unique:", len(set(captions)))
    assert isinstance(captions, list)
    assert len(captions) == num_captions
    unique_captions = len(set(captions))
    if unique_captions < num_captions / 2:
        print("Warning: Less than half of the captions are unique.")
    for caption in captions:
        assert isinstance(caption, str)
        assert len(caption) > 0

# pytest -v QAPairGen/test_caption_gen.py -k test_vllm_generate_multiple_captions -s
def test_vllm_generate_multiple_captions(client, image):
    num_captions = 10
    sampling_params = SamplingParams(temperature=0.2)
    captions = generate_multiple_captions(engine="vllm", model= "Salesforce/blip2-opt-2.7b", client=client, image=image, num_captions=num_captions, sampling_params=sampling_params)
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