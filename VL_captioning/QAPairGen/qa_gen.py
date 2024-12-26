
import time
import re
from openai import OpenAI
from vllm import SamplingParams
import concurrent.futures as futures
import warnings



# take in list of captions

# for each caption:
# pass, create word to caption mapping
# select captions with unique words (words in mapping with least number of captions)

# for each of these count captions:
# generate question answer pairs that is answerable by the caption

# return list of question answer pairs
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt1 = """
        This is an image caption: {caption}. Based only on the information in the caption, write one simple question and a one-word answer in the exact format:
        Question: <question text> Answer: <one-word answer>

        Do not include any additional information, context, or explanations.

        Now, based on the provided caption, generate your response.
        """
regex1 = r"Question:\s*(.+?)\s*Answer:\s*(\w+)"

prompt2 = """
    This is an image caption: {caption}. Based only on the information in the caption, create a JSON object containing a single key-value pair. The key should be "q" for the question, and "a" for the one-word answer. The JSON object should be in the exact format:
    {{ "q": "<question text>", "a": "<one-word answer>"}}
    Do not include any additional information, context, or explanations. Now, based on the provided caption, generate your response.
"""

regex2 = r'{\s*"q":\s*"(.+?)",\s*"a":\s*"(\w+)"\s*}'

def generate_qa_pairs_vllm(client, model, captions, count=None, sampling_params=None):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.2)
    if count is None:
        count = len(captions)
    
    print("Captions:", captions)
    # 30 QA pairs in 6 seconds
    qa_pairs = []
    def multi_threaded_chat_completion(caption):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt1.format(caption=caption)},
            ],
        }]
        return client.chat.completions.create(model=model, messages=messages, temperature=sampling_params.temperature, max_tokens=64)
    
    with futures.ThreadPoolExecutor() as executor:
        outputs = list(executor.map(multi_threaded_chat_completion, captions))

    for i in range(count):
        output = outputs[i]
        output2 = [t.message.content for t in output.choices]
        ouput_str = output2[0]
        match = re.search(r"Question:\s*(.+?)\s*Answer:\s*(\w+)", ouput_str)
        if match:
            # print(f"Match: {match.groups()}")
            question, answer = match.groups()
            qa_pairs.append((question.strip(), answer.strip()))
        else:
            warnings.warn(f"Failed to match regex for output: {ouput_str}")

    if len(qa_pairs) != min(count, len(captions)):
        warnings.warn(f"c ({len(qa_pairs)}) does not match the number of captions min({count}, {len(captions)})")
    return qa_pairs

if __name__ == "__main__":
    # Load model and tokenizer
    model = "mistralai/Mistral-7B-v0.1"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    # Example captions
    captions = [
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children in an elementary school classroom with their teacher working on a map\n",
        " a group of children and adults around the table with their hands on paper writing something\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children around the table with their teacher writing on paper in front of them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children working together on a map in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children around a table with an adult pointing to the map on it\n",
        " a group of children around the table with an adult teacher and another child pointing to something\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children sitting around the table with one girl pointing to something on paper\n",
        " a group of children and adults around the table with their hands on paper writing something\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children and adults around the table with their hands on paper writing something\n",
        " a group of students in an elementary school classroom with their teacher working on a project\n",
        " a group of children in an elementary school classroom with their teacher working on a map\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children in an elementary school classroom with their teacher working on a project\n",
        " a group of children around the table with their teacher working on something together. The image is in color\n",
        " a group of students working together on an assignment in the classroom. The teacher is standing behind them\n",
        " a group of children around the table with one girl pointing to something on paper while another child looks at her\n"
    ]

    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="idk"
    )

    tick = time.time()
    qa_pairs = generate_qa_pairs_vllm(client, model, captions, sampling_params=SamplingParams(temperature=0.2))
    print("Time taken:", time.time() - tick)
    print("Total successful QA pairs:", len(qa_pairs))
    print(qa_pairs)
