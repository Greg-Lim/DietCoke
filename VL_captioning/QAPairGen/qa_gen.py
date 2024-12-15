
import time
import caption_gen as caption_gen
import word_to_caption as word_to_caption
import re
from openai import OpenAI
from vllm import SamplingParams
import concurrent.futures as futures



# take in list of captions

# for each caption:
# pass, create word to caption mapping
# select captions with unique words (words in mapping with least number of captions)

# for each of these count captions:
# generate question answer pairs that is answerable by the caption

# return list of question answer pairs
from transformers import AutoModelForCausalLM, AutoTokenizer


# def generate_qa_pairs(llm_model, llm_tokenizer, captions):
#     qa_pairs = []
#     for caption in captions:
#         # Prompt with example for consistent format
#         prompt = f"""
#         This is an image caption: {caption}. Based only on the information in the caption, write a simple question and a one-word answer in the exact format:
#         Question: <question text> Answer: <one-word answer>

#         Do not include any additional information, context, or explanations.

#         Now, based on the provided caption, generate your response.
#         """
#         # Tokenize and prepare inputs for the model
#         inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
#         # Generate response from the model
#         generated_ids = llm_model.generate(inputs.input_ids, max_new_tokens=64, do_sample=True)
#         result = llm_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
#         # Use regex to extract all QA pairs
#         question_answer_regex = r"Question:\s*(.+?)\s*Answer:\s*(\w+)"
#         match = re.search(question_answer_regex, result)
                
#         # Append the first extracted QA pair to the list
#         if match:
#             question, answer = match.groups()
#             qa_pairs.append((question.strip(), answer.strip()))
    
#     return qa_pairs

prompt = """
        This is an image caption: {caption}. Based only on the information in the caption, write one simple question and a one-word answer in the exact format:
        Question: <question text> Answer: <one-word answer>

        Do not include any additional information, context, or explanations.

        Now, based on the provided caption, generate your response.
        """

def generate_qa_pairs(engine, captions, **kwargs):
    if engine == "huggingface":
        return _generate_qa_pairs_hf(captions=captions, **kwargs)
    elif engine == "vllm":
        return _generate_qa_pairs_vllm(captions=captions, **kwargs)
    
def _generate_qa_pairs_hf(model, tokenizer, captions):
    qa_pairs = []
    for caption in captions:
        prompt = f"""
        This is an image caption: {caption}. Based only on the information in the caption, write a simple question and a one-word answer in the exact format:
        Question: <question text> Answer: <one-word answer>

        Do not include any additional information, context, or explanations.

        Now, based on the provided caption, generate your response.
        """
        # Tokenize and prepare inputs for the model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Generate response from the model
        generated_ids = model.generate(inputs.input_ids, max_new_tokens=64, do_sample=True)
        result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Use regex to extract all QA pairs
        question_answer_regex = r"Question:\s*(.+?)\s*Answer:\s*(\w+)"
        match = re.search(question_answer_regex, result)
                
        # Append the first extracted QA pair to the list
        if match:
            question, answer = match.groups()
            qa_pairs.append((question.strip(), answer.strip()))
    
    return qa_pairs

def _generate_qa_pairs_vllm(client, model, captions, sampling_params=None):
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.2)
    return _generate_qa_pairs_vllm_chat_completion_multithreading(client, model, captions, sampling_params)
    # return _generate_qa_pairs_vllm_batch(client, model, captions, sampling_params)

def _generate_qa_pairs_vllm_chat_completion_linear(client, model, captions, sampling_params):
    # 30 QA in 30 seconds
    qa_pairs = []
    for caption in captions:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.format(caption=caption)},
            ],
        }]

        outputs = client.chat.completions.create(model=model, messages=messages, temperature=sampling_params.temperature, max_tokens=64)
        outputs = [output.message.content for output in outputs.choices]
        for output in outputs:
            match = re.search(r"Question:\s*(.+?)\s*Answer:\s*(\w+)", output)
            if match:
                question, answer = match.groups()
                qa_pairs.append((question.strip(), answer.strip()))
    return qa_pairs


def _generate_qa_pairs_vllm_chat_completion_multithreading(client, model, captions, sampling_params):
    # 30 QA pairs in 6 seconds
    qa_pairs = []
    def multi_threaded_chat_completion(caption):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.format(caption=caption)},
            ],
        }]
        return client.chat.completions.create(model=model, messages=messages, temperature=sampling_params.temperature, max_tokens=64)
    
    with futures.ThreadPoolExecutor() as executor:
        outputs = list(executor.map(multi_threaded_chat_completion, captions))

    for output in outputs:
        outputs = [output.message.content for output in output.choices]
        for output in outputs:
            match = re.search(r"Question:\s*(.+?)\s*Answer:\s*(\w+)", output)
            if match:
                question, answer = match.groups()
                qa_pairs.append((question.strip(), answer.strip()))
    return qa_pairs

def _generate_qa_pairs_vllm_batch(client, model, caption, sampling_params):
    # this seems not possible with the vLLM API
    qa_pairs = []
    request_input_objects = [] # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]}}

    for caption in captions:
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": prompt.format(caption=caption)},
            ],
        }]
        request_input_objects.append({"model": model, "messages": messages, "temperature": sampling_params.temperature})

    batch_input_file = client.files.create(
        file = request_input_objects,
        purpose = "batch"
    )
    outputs = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="10s",
        metadata={
            "description": "batch job for generating QA pairs"
        }
    )
    print(outputs)



if __name__ == "__main__":
    # Load model and tokenizer

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
    # Generate QA pairs
    # qa_pairs = generate_qa_pairs(model, tokenizer, captions)
    if 0 and "testing hf":
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        qa_pairs = generate_qa_pairs("huggingface", captions, model= model, tokenizer= tokenizer)

    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="idk"
    )

    tick = time.time()
    qa_pairs = generate_qa_pairs("vllm", captions, client=client, model="mistralai/Mistral-7B-v0.1", sampling_params=SamplingParams(temperature=0.2))
    print("Time taken vllm:", time.time() - tick)
    print(qa_pairs)
