from concurrent import futures
import time
from PIL import Image
import requests
import base64
from io import BytesIO

target = "http://localhost:8000"

url = "https://solutions.cal.org/wp-content/uploads/2022/06/solutions-oral-proficiency-oct2019.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
question = "What is behind the people in the image?"

def test_read_root():
    response = requests.get(f"{target}/")
    assert response.status_code == 200


def test_generate_caption_qa():
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    files = {
        "file": ("image.jpg", image_bytes, "image/jpeg"),
    }
    data = {
        "question": question,
    }

    response = requests.post(f"{target}/generate_caption_qa", files=files, data=data)

    response.raise_for_status()

    print(response.json())

def triple_test_generate_caption_qa():
    with futures.ThreadPoolExecutor() as executor:
            future_1 = executor.submit(test_generate_caption_qa)
            future_2 = executor.submit(test_generate_caption_qa)
            future_3 = executor.submit(test_generate_caption_qa)
            future_1.result()
            future_2.result()
            future_3.result()

if __name__ == "__main__":
    test_read_root()
    start = time.time()
    test_generate_caption_qa()
    t_single = time.time() - start
    
    # start = time.time()
    # triple_test_generate_caption_qa()
    # t_triple = time.time() - start
    print("Single:", t_single)
    # print("Triple:", t_triple)