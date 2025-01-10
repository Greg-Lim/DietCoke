
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import logging
import utils
from dataset.utils import save_result
import json
import random
import openai
import en_core_web_sm
import time
import concurrent.futures as futures
import os
import ruamel.yaml as yaml
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from transformers import AutoTokenizer

from QAPairGen import caption_gen, qa_gen, word_to_caption

nlp = en_core_web_sm.load()

def llm_generate(client, **kwargs):
    return client.chat.completions.create(**kwargs)

def create_cap_dic(caption_data):
    # Not needed, it reads from the file and remaps the data
    cap = []
    que_id = []
    for i in caption_data:
        que_id.append(i['question_id'])
        if isinstance(i['caption'], list):
            total_caption_list = []
            for ctx_id, cap_ in enumerate(i['caption'][:100]):
                total_caption_list.append((cap_.capitalize().strip()).rstrip()+".")
            cap.append(total_caption_list)
        else:
            raise NotImplementedError()
    caption_dict = dict(zip(que_id, cap))
    return caption_dict

def create_ans_to_cap_dic(ans_to_cap_data):
    # Not needed, it reads from the file and remaps the data
    que_id = []
    ans_dicts = []

    for i in ans_to_cap_data:
        que_id.append(i['question_id'])
        if 'ans_to_cap_dict' not in i.keys():
            key = 'tag'
        else:
            key = 'ans_to_cap_dict'
        if isinstance(i[key], dict):
                ans_dicts.append(i[key])
        else:
            raise NotImplementedError()
    ans_to_cap_dicts = dict(zip(que_id, ans_dicts))
    return ans_to_cap_dicts

def create_generated_question_dic(question_data):
    # Not needed, it reads from the file and remaps the data
    que_id = []
    syn_question = []
    syn_answer = []
    que_id = []
    ans_dicts = []

    for i in question_data:
        que_id.append(i['question_id'])
        if isinstance(i['question'], list):
            total_syn_question_list = []
            for ctx_id, syn_question_ in enumerate(i['question']):
                total_syn_question_list.append(syn_question_.capitalize().strip().rstrip())
            syn_question.append(total_syn_question_list)
        else:
            raise NotImplementedError()
        if isinstance(i['answer'], list):
            total_syn_answer_list = []
            for ctx_id, syn_answer_ in enumerate(i['answer']):
                total_syn_answer_list.append(syn_answer_.capitalize().strip().rstrip())
            syn_answer.append(total_syn_answer_list)
        else:
            raise NotImplementedError()
    syn_question_dict = dict(zip(que_id, syn_question))
    syn_answer_dict = dict(zip(que_id, syn_answer))

    return syn_question_dict,syn_answer_dict

def extract_string(text):

    dot_index = text.find(".")
    newline_index = text.find("\n")
    parenthesis_index = text.find(" (")
    comma_index = text.find(",")
    index_or = text.find(" or ")


    indices = [index for index in [dot_index, newline_index, parenthesis_index, comma_index, index_or] if index != -1]
    if indices:
        index = min(indices)
    else:
        index = -1


    if index != -1:
        return text[:index]
    else:
        return text

def remove_prefix(word):

    if word.startswith("a "):
        return word[2:]

    elif word.startswith("an "):
        return word[3:]
    elif word.startswith("the "):
        return word[4:]
    else:
        return word

# for OK-VQA
def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    tagged_word = nltk.pos_tag([word])
    pos = tagged_word[0][1][0].lower()

    if pos in ['v', 'n']:
        if word.endswith("ing"):
            return lemmatizer.lemmatize(word, pos='v')
        return lemmatizer.lemmatize(word, pos=pos)
    else:
        return lemmatizer.lemmatize(word)

def post_process(pred_answer):
    pred_answer = extract_string(pred_answer)
    pred_answer = remove_prefix(pred_answer)
    # words_list = pred_answer.split()  # for OK-VQA
    # words_list_new = [lemmatize_word(w) for w in words_list] # for OK-VQA
    # pred_answer = " ".join(words_list_new) # for OK-VQA
    return pred_answer

class Knowledge():

    def create_context_prompt(ans_dict_queid,syn_ans_queid,caption,config):
        Context_Prompt = ""
        mycontexts_id = []
        for idx in range(config['num_caps_per_img']):
            if config['dataset'] in ['vqa','vqasubset','vqatest']:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(), [0])
            else:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(),[0])  ## rare_answers, each answer can occur in multiple captions,so it is a caption list
            for cap_id in cap_id_list:
                if cap_id not in mycontexts_id:
                    Context_Prompt += caption[cap_id]
                    mycontexts_id.append(cap_id)
                    break  # We just take one cap for each answer
        return Context_Prompt

    def create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config):
        Task_Prompt  = ""
        for idx in range(config['num_question_per_img']):
            if config['random_question']:
                qa_idx = random.randint(0, len(syn_question_queid) - 1)
            else:
                qa_idx = idx
            if config['dataset'] in ['vqa', 'vqasubset', 'vqatest'] and config['question_type'] != 'rule' and config[
                    'num_question_per_img'] > 0 and idx < 1:  ## yes and no questions for vqav2
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid_next[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:no\n"
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += "yes\n"
                    Task_Prompt += "Question:Is this a toilet?\n"
                    Task_Prompt += "Answer:no\n"
            if config['question_type'] == 'rule':   # Rule-Based Question Generation
                Noun_Questions = ["What item is this in this picture?",
                                "What item is that in this picture?"]

                Verb_Questions = ["What action is being done in this picture?",
                                "Why is this item doing in this picture?",
                                "Which action is being taken in this picture?",
                                "What action is item doing in this picture?",
                                "What action is item performing in this picture?"]

                Adj_Questions = ["How to describe one item in this picture?",
                                "What is item's ADJ TYPE in this picture?",
                                "What is the ADJ TYPE in this picture?"]

                Task_Prompt += "Question:"
                doc = nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                if doc[-1].pos_ == "NOUN":
                    Task_Prompt += Noun_Questions[random.randint(0, len(Noun_Questions) - 1)]
                elif doc[-1].pos_ == "VERB":
                    Task_Prompt += Verb_Questions[random.randint(0, len(Verb_Questions) - 1)]
                elif doc[-1].pos_ == "ADJ":
                    Task_Prompt += Adj_Questions[random.randint(0, len(Adj_Questions) - 1)]

                Task_Prompt += '\n'

                Task_Prompt += "Answer:"
                Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                Task_Prompt += '\n'
            else:
                if len(syn_ans_queid[qa_idx % len(syn_ans_queid)].split()) < 5:
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[(qa_idx) % len(syn_question_queid)]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                    Task_Prompt += '\n'

        # print(Task_Prompt)
        return Task_Prompt

    def create_messages(Context_Prompt, Task_Prompt, question):
        long_knowledge_messages = []
        short_knowledge_messages = []

        contents = []
        contents.append("You are going to answer questions according to the contexts:" + Context_Prompt)
        contents.append("Ok, please go ahead and ask your question.")
        Task_Prompt = Task_Prompt.split('\n')
        for example in Task_Prompt:
            pos = example.find(':')
            contents.append(example[pos+1:])
        if len(contents) % 2 == 1: x = contents.pop()
        counter = 1
        for content in contents:
            if counter % 2 == 1:
                long_knowledge_messages.append({"role": "user", "content": content})
                short_knowledge_messages.append({"role": "user", "content": content})
            else:
                long_knowledge_messages.append({"role": "assistant", "content": content})
                short_knowledge_messages.append({"role": "assistant", "content": content})
            counter += 1
        long_knowledge_messages.append({"role": "user", "content": question})
        short_knowledge_messages.append({"role": "user", "content": question})
        long_knowledge_messages.append({"role": "assistant", "content": "I don't have enough knowledge to answer this question."})
        short_knowledge_messages.append({"role": "assistant", "content": "I don't have enough knowledge to answer this question."})
        long_knowledge_messages.append({"role": "user", "content": "Please provide background knowledge related to this question"})
        short_knowledge_messages.append({"role": "user", "content": "Please provide background knowledge related to this question in a single sentence."})
        return long_knowledge_messages, short_knowledge_messages

    def knowledge_generate(model, test_data, caption_dict,syn_question_dict,syn_answer_dict,ans_to_cap_dicts, config):
        results_long = {}
        results_short = {}
        test_data = test_data[:2]
        for n, per_test_data in enumerate(test_data):
            question = per_test_data['question'].lower().strip()
            question_id = per_test_data['question_id']
            ans_dict_queid = ans_to_cap_dicts[question_id]
            question_id_next = test_data[(n + 1) % len(test_data)]['question_id']
            syn_question_queid = syn_question_dict[question_id]
            syn_question_queid_next = syn_question_dict[question_id_next]
            caption = caption_dict[question_id]
            syn_ans_queid = syn_answer_dict[question_id]

            Task_Prompt = Knowledge.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)
            Context_Prompt = Knowledge.create_context_prompt(ans_dict_queid, syn_ans_queid, caption, config)

            long_knowledge_messages, short_knowledge_messages = Knowledge.create_messages(Context_Prompt, Task_Prompt, question)

            result_short = llm_generate(model=model, messages=short_knowledge_messages, temperature=0.7, max_tokens=1000)
            result_long = llm_generate(model=model, messages=long_knowledge_messages, temperature=0.7, max_tokens=1000)

            full_text_short = result_short['choices'][0]['message']['content']
            short_knowledge = full_text_short.split("[/INST]")[-1]
            short_knowledge = short_knowledge.split("</s>")[0]
            short_knowledge = short_knowledge.replace("\n", " ")

            full_text_long = result_long['choices'][0]['message']['content']
            long_knowledge = full_text_long.split("[/INST]")[-1]
            long_knowledge = long_knowledge.split("</s>")[0]
            long_knowledge = long_knowledge.replace("\n", " ")

            print("long ", long_knowledge)
            print("short ", short_knowledge)

            print(f"ques_id: {question_id}  question: {question} gt_answer: {per_test_data['answer']} short_knowledge: {short_knowledge} long_knowledge: {long_knowledge} captions: {Context_Prompt}\n")
            results_long[question_id] = long_knowledge
            results_short[question_id] = short_knowledge
        return results_long, results_short
    
    def knowledge_generate_single(llm_client, question, ans_dict_queid, syn_question_queid, caption, syn_ans_queid, config):
        """
        Generates long and short knowledge responses based on the provided inputs.

        Args:
            llm_client: The language model used for generating responses.
            question (str): The input question for which knowledge is to be generated.
            ans_dict_queid (dict): Dictionary of answers mapped to question IDs, generated by map_words_to_captions.
            syn_question_queid (str): Synthetic question ID generated by generate_qa_pairs.
            caption (str): Caption generated by generate_multiple_captions.
            syn_ans_queid (str): Synthetic answer ID generated by generate_qa_pairs.
            config (dict): Configuration settings for generating prompts and responses.

        Returns: (long_knowledge, short_knowledge)
        """
        question
        # question_id = per_test_data['question_id']
        ans_dict_queid # generated by map_words_to_captions
        # question_id_next = test_data[(n + 1) % len(test_data)]['question_id'] # not needed (this uses a different question as a example)
        syn_question_queid  # generate by generate_qa_pairs
        syn_question_queid_next = None # just for a input to use the old function
        caption # caption generated by generate_multiple_captions
        syn_ans_queid # generated by generate_qa_pairs

        Task_Prompt = Knowledge.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)
        Context_Prompt = Knowledge.create_context_prompt(ans_dict_queid, syn_ans_queid, caption, config)

        long_knowledge_messages, short_knowledge_messages = Knowledge.create_messages(Context_Prompt, Task_Prompt, question)

        with futures.ThreadPoolExecutor() as executor:
            future_short = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=short_knowledge_messages, temperature=0.7, max_tokens=1000)
            future_long = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=long_knowledge_messages, temperature=0.7, max_tokens=1000)
            result_short = future_short.result()
            result_long = future_long.result()

        full_text_short = result_short.choices[0].message.content
        short_knowledge = full_text_short.split("[/INST]")[-1]
        short_knowledge = short_knowledge.split("</s>")[0]
        short_knowledge = short_knowledge.replace("\n", " ")

        full_text_long = result_long.choices[0].message.content
        long_knowledge = full_text_long.split("[/INST]")[-1]
        long_knowledge = long_knowledge.split("</s>")[0]
        long_knowledge = long_knowledge.replace("\n", " ")

        print("long ", long_knowledge)
        print("short ", short_knowledge)

        return long_knowledge, short_knowledge

class AnswerCandidates():

    def create_context_prompt(ans_dict_queid,syn_ans_queid,caption,config):
        Context_Prompt = ""
        mycontexts_id = []
        for idx in range(config['num_caps_per_img']):
            if config['dataset'] in ['vqa','vqasubset','vqatest']:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(), [0])
            else:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(),[0])  ## rare_answers, each answer can occur in multiple captions,so it is a caption list
            for cap_id in cap_id_list:
                if cap_id not in mycontexts_id:
                    Context_Prompt += caption[cap_id]
                    mycontexts_id.append(cap_id)
                    break  # We just take one cap for each answer
        return Context_Prompt

    def create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config):
        Task_Prompt  = ""
        for idx in range(config['num_question_per_img']):
            if config['random_question']:
                qa_idx = random.randint(0, len(syn_question_queid) - 1)
            else:
                qa_idx = idx
            if config['dataset'] in ['vqa', 'vqasubset', 'vqatest'] and config['question_type'] != 'rule' and config[
                    'num_question_per_img'] > 0 and idx < 1:  ## yes and no questions for vqav2
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid_next[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:no\n"
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += "yes\n"
                    Task_Prompt += "Question:Is this a toilet?\n"
                    Task_Prompt += "Answer:no\n"
            if config['question_type'] == 'rule':   # Rule-Based Question Generation
                Noun_Questions = ["What item is this in this picture?",
                                "What item is that in this picture?"]

                Verb_Questions = ["What action is being done in this picture?",
                                "Why is this item doing in this picture?",
                                "Which action is being taken in this picture?",
                                "What action is item doing in this picture?",
                                "What action is item performing in this picture?"]

                Adj_Questions = ["How to describe one item in this picture?",
                                "What is item's ADJ TYPE in this picture?",
                                "What is the ADJ TYPE in this picture?"]

                Task_Prompt += "Question:"
                doc = nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                if doc[-1].pos_ == "NOUN":
                    Task_Prompt += Noun_Questions[random.randint(0, len(Noun_Questions) - 1)]
                elif doc[-1].pos_ == "VERB":
                    Task_Prompt += Verb_Questions[random.randint(0, len(Verb_Questions) - 1)]
                elif doc[-1].pos_ == "ADJ":
                    Task_Prompt += Adj_Questions[random.randint(0, len(Adj_Questions) - 1)]

                Task_Prompt += '\n'

                Task_Prompt += "Answer:"
                Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                Task_Prompt += '\n'
            else:
                if len(syn_ans_queid[qa_idx % len(syn_ans_queid)].split()) < 5:
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[(qa_idx) % len(syn_question_queid)]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                    Task_Prompt += '\n'


        # print(Task_Prompt)
        return Task_Prompt

    def create_messages(Prompt, Context_Prompt, Task_Prompt, question):
        messages = []

        contents = []
        contents.append(Prompt + Context_Prompt)
        contents.append("Ok, please go ahead and ask your question.")
        Task_Prompt = Task_Prompt.split('\n')
        for example in Task_Prompt:
            pos = example.find(':')
            contents.append(example[pos+1:])
        if len(contents) % 2 == 1: x = contents.pop()
        counter = 1
        for content in contents:
            if counter % 2 == 1:
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
            counter += 1
        messages.append({"role": "user", "content": question + "Try to answer with only one word."})
        return messages

    def answer_candidates_generate(model, test_data, caption_dict, syn_question_dict, syn_answer_dict, ans_to_cap_dicts, config, long_knowledge, short_knowledge):
        results_long = []
        results_short = []
        results_cap = []
        test_data = test_data[:2]
        print("in answer, short:")
        print(short_knowledge)
        for n, per_test_data in enumerate(test_data):
            question = per_test_data['question'].lower().strip()
            question_id = per_test_data['question_id']

            print(question_id)

            question_id_next = test_data[(n+1)%len(test_data)]['question_id']
            ans_dict_queid = ans_to_cap_dicts[question_id]
            syn_question_queid = syn_question_dict[question_id]
            syn_question_queid_next = syn_question_dict[question_id_next]
            caption = caption_dict[question_id]
            syn_ans_queid = syn_answer_dict[question_id]

            Context_Prompt = AnswerCandidates.create_context_prompt(ans_dict_queid, syn_ans_queid, caption, config)
            Long_Knowledge_Prompt = long_knowledge[question_id]
            Short_Knowledge_Prompt = short_knowledge[question_id]

            Task_Prompt = AnswerCandidates.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)
            Prompt = "Please answer the questions with only one word according to the given background knowledge.\n"
            Img2PromptLong = "Knowledge:" + Long_Knowledge_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
            Img2PromptShort = "Knowledge:" + Short_Knowledge_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
            Img2PromptCap = "Context:" + Context_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
            PromptLong = Prompt + Img2PromptLong
            PromptShort = Prompt + Img2PromptShort
            PromptCap = Prompt + Img2PromptCap

            message_long = [{'role':'user', 'content': PromptLong}]
            message_short = [{'role':'user', 'content': PromptShort}]
            message_cap = [{'role':'user', 'content': PromptCap}]

            pred_answer_long = llm_generate(model=model, messages=message_long, temperature=0, max_tokens=20)
            pred_answer_long = pred_answer_long['choices'][0]['message']['content']
            pred_answer_long = pred_answer_long.lower()
            pred_answer_long = post_process(pred_answer_long)

            pred_answer_short = llm_generate(model=model, messages=message_short, temperature=0, max_tokens=20)
            pred_answer_short = pred_answer_short['choices'][0]['message']['content']
            pred_answer_short = pred_answer_short.lower()
            pred_answer_short = post_process(pred_answer_long)

            pred_answer_cap = llm_generate(model=model, messages=message_cap, temperature=0, max_tokens=20)
            pred_answer_cap = pred_answer_cap['choices'][0]['message']['content']
            pred_answer_cap = pred_answer_cap.lower()
            pred_answer_cap = post_process(pred_answer_long)

            print({"question": question, "pred_answer_long": pred_answer_long, "pred_answer_short": pred_answer_short, "pred_answer_cap": pred_answer_cap, "gt_answer": per_test_data['answer']})
            print("\n")
            results_long.append({"question_id": question_id, "answer": pred_answer_long})
            results_short.append({"question_id": question_id, "answer": pred_answer_short})
            results_cap.append({"question_id": question_id, "answer": pred_answer_cap})
        return results_long, results_short, results_cap
    
    def answer_candidates_generate_single(llm_client, question, ans_dict_queid, syn_question_queid, caption, syn_ans_queid, config, long_knowledge, short_knowledge):
        """
        Parameters:
        llm_client (object): The language model used to generate answers.
        question (str): The question for which answers are to be generated.
        ans_dict_queid (dict): Dictionary mapping question IDs to answers.
        syn_question_queid (str): Synthetic question ID generated by generate_qa_pairs.
        caption (str): Caption generated by generate_multiple_captions.
        syn_ans_queid (str): Synthetic answer ID generated by generate_qa_pairs.
        config (dict): Configuration dictionary containing various settings.
        long_knowledge (str): Long background knowledge text.
        short_knowledge (str): Short background knowledge text.
        Returns:
        tuple: A tuple containing three generated answers (pred_answer_long, pred_answer_short, pred_answer_cap).
        """
        
        question
        # question_id = per_test_data['question_id'] # not needed
        # question_id_next = test_data[(n+1)%len(test_data)]['question_id'] # not needed (this uses a different question as a example)
        ans_dict_queid # generated by map_words_to_captions
        syn_question_queid # generate by generate_qa_pairs
        syn_question_queid_next = None # not needed (this uses a different question as a example)
        caption # caption generated by generate_multiple_captions
        syn_ans_queid # generated by generate_qa_pairs

        Context_Prompt = AnswerCandidates.create_context_prompt(ans_dict_queid, syn_ans_queid, caption, config)
        Long_Knowledge_Prompt = long_knowledge
        Short_Knowledge_Prompt = short_knowledge

        Task_Prompt = AnswerCandidates.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)
        Prompt = "Please answer the questions with only one word according to the given background knowledge.\n"
        Img2PromptLong = "Knowledge:" + Long_Knowledge_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
        Img2PromptShort = "Knowledge:" + Short_Knowledge_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
        Img2PromptCap = "Context:" + Context_Prompt + "\n" + Task_Prompt + "Question:" + question + "\nAnswer:"
        PromptLong = Prompt + Img2PromptLong
        PromptShort = Prompt + Img2PromptShort
        PromptCap = Prompt + Img2PromptCap

        message_long = [{'role':'user', 'content': PromptLong}]
        message_short = [{'role':'user', 'content': PromptShort}]
        message_cap = [{'role':'user', 'content': PromptCap}]

        with futures.ThreadPoolExecutor() as executor:
            future_long = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=message_long, temperature=0, max_tokens=20)
            future_short = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=message_short, temperature=0, max_tokens=20)
            future_cap = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=message_cap, temperature=0, max_tokens=20)
            
            pred_answer_long = future_long.result()
            pred_answer_short = future_short.result()
            pred_answer_cap = future_cap.result()

        pred_answer_long = pred_answer_long.choices[0].message.content
        pred_answer_long = pred_answer_long.lower()
        pred_answer_long = post_process(pred_answer_long)

        pred_answer_short = pred_answer_short.choices[0].message.content
        pred_answer_short = pred_answer_short.lower()
        pred_answer_short = post_process(pred_answer_short)

        pred_answer_cap = pred_answer_cap.choices[0].message.content
        pred_answer_cap = pred_answer_cap.lower()
        pred_answer_cap = post_process(pred_answer_cap)

        print({"question": question, "pred_answer_long": pred_answer_long, "pred_answer_short": pred_answer_short, "pred_answer_cap": pred_answer_cap})
        print("\n")
        return pred_answer_long, pred_answer_short, pred_answer_cap     
        

class AutoRationales():

    def create_context_prompt(ans_dict_queid,syn_ans_queid,caption,config):
        Context_Prompt = ""
        mycontexts_id = []
        for idx in range(config['num_caps_per_img']):
            if config['dataset'] in ['vqa','vqasubset','vqatest']:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(), [0])
            else:
                cap_id_list = ans_dict_queid.get(
                    syn_ans_queid[(len(syn_ans_queid) - 1 - idx) % len(syn_ans_queid)][:-1].lower(),[0])  ## rare_answers, each answer can occur in multiple captions,so it is a caption list
            for cap_id in cap_id_list:
                if cap_id not in mycontexts_id:
                    Context_Prompt += caption[cap_id]
                    mycontexts_id.append(cap_id)
                    break  # We just take one cap for each answer
        return Context_Prompt

    def create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config):
        Task_Prompt  = ""
        for idx in range(config['num_question_per_img']):
            if config['random_question']:
                qa_idx = random.randint(0, len(syn_question_queid) - 1)
            else:
                qa_idx = idx
            if config['dataset'] in ['vqa', 'vqasubset', 'vqatest'] and config['question_type'] != 'rule' and config[
                    'num_question_per_img'] > 0 and idx < 1:  ## yes and no questions for vqav2
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid_next[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:no\n"
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += "yes\n"
                    Task_Prompt += "Question:Is this a toilet?\n"
                    Task_Prompt += "Answer:no\n"
            if config['question_type'] == 'rule':   # Rule-Based Question Generation
                Noun_Questions = ["What item is this in this picture?",
                                "What item is that in this picture?"]

                Verb_Questions = ["What action is being done in this picture?",
                                "Why is this item doing in this picture?",
                                "Which action is being taken in this picture?",
                                "What action is item doing in this picture?",
                                "What action is item performing in this picture?"]

                Adj_Questions = ["How to describe one item in this picture?",
                                "What is item's ADJ TYPE in this picture?",
                                "What is the ADJ TYPE in this picture?"]

                Task_Prompt += "Question:"
                doc = nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                if doc[-1].pos_ == "NOUN":
                    Task_Prompt += Noun_Questions[random.randint(0, len(Noun_Questions) - 1)]
                elif doc[-1].pos_ == "VERB":
                    Task_Prompt += Verb_Questions[random.randint(0, len(Verb_Questions) - 1)]
                elif doc[-1].pos_ == "ADJ":
                    Task_Prompt += Adj_Questions[random.randint(0, len(Adj_Questions) - 1)]

                Task_Prompt += '\n'

                Task_Prompt += "Answer:"
                Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                Task_Prompt += '\n'
            else:
                if len(syn_ans_queid[qa_idx % len(syn_ans_queid)].split()) < 5:
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[(qa_idx) % len(syn_question_queid)]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                    Task_Prompt += '\n'


        # print(Task_Prompt)
        return Task_Prompt

    def create_messages(Prompt, Context_Prompt, Task_Prompt, question, answer):
        messages = []

        contents = []
        contents.append(Prompt + Context_Prompt)
        contents.append("Ok, please go ahead and ask your question.")
        Task_Prompt = Task_Prompt.split('\n')
        for example in Task_Prompt:
            pos = example.find(':')
            contents.append(example[pos+1:])
        if len(contents) % 2 == 1: x = contents.pop()
        counter = 1
        for content in contents:
            if counter % 2 == 1:
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
            counter += 1
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": "Please explain the reasoning behind your answer in a single sentence."})
        return messages

    def auto_rationale_generate(model, test_data, caption_dict, syn_question_dict, syn_answer_dict, ans_to_cap_dicts, config, answer_long, answer_short, answer_cap):
        results_long = {}
        results_short = {}
        results_cap = {}
        test_data = test_data[:2]
        index = 0
        for n, per_test_data in enumerate(test_data):
            question = per_test_data['question'].lower().strip()
            question_id = per_test_data['question_id']
            cur_answer_long = answer_long[index]['answer']
            cur_answer_short = answer_short[index]['answer']
            cur_answer_cap = answer_cap[index]['answer']
            index += 1
            question_id_next = test_data[(n+1)%len(test_data)]['question_id']
            ans_dict_queid = ans_to_cap_dicts[question_id]
            syn_question_queid = syn_question_dict[question_id]
            syn_question_queid_next = syn_question_dict[question_id_next]
            caption = caption_dict[question_id]
            syn_ans_queid = syn_answer_dict[question_id]
            Context_Prompt = AutoRationales.create_context_prompt(ans_dict_queid,syn_ans_queid,caption,config)
            Task_Prompt = AutoRationales.create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config)

            Prompt = "You are going to answer questions according to the contexts:"
            messages_long = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_long)
            messages_short = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_short)
            messages_cap = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_cap)

            rationale_long = llm_generate(model=model, messages=messages_long, temperature=0, max_tokens=1000)
            rationale_long = rationale_long['choices'][0]['message']['content']
            rationale_long = rationale_long.split("[/INST]")[-1]
            rationale_long = rationale_long.split("</s>")[0]
            rationale_long = rationale_long.replace('\n', ' ')

            rationale_short = llm_generate(model=model, messages=messages_short, temperature=0, max_tokens=1000)
            rationale_short = rationale_short['choices'][0]['message']['content']
            rationale_short = rationale_short.split("[/INST]")[-1]
            rationale_short = rationale_short.split("</s>")[0]
            rationale_short = rationale_short.replace('\n', ' ')

            rationale_cap = llm_generate(model=model, messages=messages_cap, temperature=0, max_tokens=1000)
            rationale_cap = rationale_cap['choices'][0]['message']['content']
            rationale_cap = rationale_cap.split("[/INST]")[-1]
            rationale_cap = rationale_cap.split("</s>")[0]
            rationale_cap = rationale_cap.replace('\n', ' ')

            print(f"ques_id: {question_id}  question: {question}  context: {Context_Prompt}\n")
            print(f"pred_answer_long: {cur_answer_long} rationale_long: {rationale_long}\n")
            print(f"pred_answer_short: {cur_answer_short} rationale_short: {rationale_short}\n")
            print(f"pred_answer_cap: {cur_answer_cap} rationale_cap: {rationale_cap}\n")
            results_long[str(question_id)] = rationale_long
            results_short[str(question_id)] = rationale_short
            results_cap[str(question_id)] = rationale_cap
        return results_long, results_short, results_cap
    
    def auto_rationale_generate_single(llm_client, question, answer_long, answer_short, answer_cap, ans_dict_queid, syn_question_queid, caption, syn_ans_queid, config):
        
        question 
        # question_id = per_test_data['question_id'] # not needed
        cur_answer_long = answer_long
        cur_answer_short = answer_short
        cur_answer_cap = answer_cap
        # index += 1
        # question_id_next = test_data[(n+1)%len(test_data)]['question_id'] # not needed (this uses a different question as a example)
        ans_dict_queid # generated by map_words_to_captions
        syn_question_queid # generate by generate_qa_pairs
        # syn_question_queid_next = syn_question_dict[question_id_next] # not needed (this uses a different question as a example)
        syn_question_queid_next = None # just for a input to use the old function
        caption
        syn_ans_queid # generated by generate_qa_pairs
        Context_Prompt = AutoRationales.create_context_prompt(ans_dict_queid,syn_ans_queid,caption,config)
        Task_Prompt = AutoRationales.create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config)

        Prompt = "You are going to answer questions according to the contexts:"
        messages_long = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_long)
        messages_short = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_short)
        messages_cap = AutoRationales.create_messages(Prompt, Context_Prompt, Task_Prompt, question, cur_answer_cap)

        with futures.ThreadPoolExecutor() as executor:
            future_long = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=messages_long, temperature=0, max_tokens=1000)
            future_short = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=messages_short, temperature=0, max_tokens=1000)
            future_cap = executor.submit(llm_generate, client=llm_client, model=config["llm_model"], messages=messages_cap, temperature=0, max_tokens=1000)
            
            rationale_long = future_long.result()
            rationale_short = future_short.result()
            rationale_cap = future_cap.result()

        rationale_long = rationale_long.choices[0].message.content
        rationale_long = rationale_long.split("[/INST]")[-1]
        rationale_long = rationale_long.split("</s>")[0]
        rationale_long = rationale_long.replace('\n', ' ')

        rationale_short = rationale_short.choices[0].message.content
        rationale_short = rationale_short.split("[/INST]")[-1]
        rationale_short = rationale_short.split("</s>")[0]
        rationale_short = rationale_short.replace('\n', ' ')

        rationale_cap = rationale_cap.choices[0].message.content
        rationale_cap = rationale_cap.split("[/INST]")[-1]
        rationale_cap = rationale_cap.split("</s>")[0]
        rationale_cap = rationale_cap.replace('\n', ' ')

        print(f"question: {question}  context: {Context_Prompt}\n")
        print(f"pred_answer_long: {cur_answer_long} rationale_long: {rationale_long}\n")
        print(f"pred_answer_short: {cur_answer_short} rationale_short: {rationale_short}\n")
        print(f"pred_answer_cap: {cur_answer_cap} rationale_cap: {rationale_cap}\n")
        return rationale_long, rationale_short, rationale_cap                      
        

class AnswerFusion():

    def get_string_before_first_period(text):
        period_index = text.find('.')
        if period_index != -1:
            return text[:period_index]
        return text

    def create_task_prompt(syn_question_queid,syn_ans_queid,syn_question_queid_next,config):
        Task_Prompt  = ""
        for idx in range(config['num_question_per_img']):
            if config['random_question']:
                qa_idx = random.randint(0, len(syn_question_queid) - 1)
            else:
                qa_idx = idx
            if config['dataset'] in ['vqa', 'vqasubset', 'vqatest'] and config['question_type'] != 'rule' and config[
                    'num_question_per_img'] > 0 and idx < 1:  ## yes and no questions for vqav2
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid_next[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:no\n"
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[-1]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += "yes\n"
                    Task_Prompt += "Question:Is this a toilet?\n"
                    Task_Prompt += "Answer:no\n"
            if config['question_type'] == 'rule':   # Rule-Based Question Generation
                Noun_Questions = ["What item is this in this picture?",
                                "What item is that in this picture?"]

                Verb_Questions = ["What action is being done in this picture?",
                                "Why is this item doing in this picture?",
                                "Which action is being taken in this picture?",
                                "What action is item doing in this picture?",
                                "What action is item performing in this picture?"]

                Adj_Questions = ["How to describe one item in this picture?",
                                "What is item's ADJ TYPE in this picture?",
                                "What is the ADJ TYPE in this picture?"]

                Task_Prompt += "Question:"
                doc = nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                if doc[-1].pos_ == "NOUN":
                    Task_Prompt += Noun_Questions[random.randint(0, len(Noun_Questions) - 1)]
                elif doc[-1].pos_ == "VERB":
                    Task_Prompt += Verb_Questions[random.randint(0, len(Verb_Questions) - 1)]
                elif doc[-1].pos_ == "ADJ":
                    Task_Prompt += Adj_Questions[random.randint(0, len(Adj_Questions) - 1)]

                Task_Prompt += '\n'

                Task_Prompt += "Answer:"
                Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                Task_Prompt += '\n'
            else:
                if len(syn_ans_queid[qa_idx % len(syn_ans_queid)].split()) < 5:
                    Task_Prompt += "Question:"
                    Task_Prompt += syn_question_queid[(qa_idx) % len(syn_question_queid)]
                    Task_Prompt += '\n'
                    Task_Prompt += "Answer:"
                    Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                    Task_Prompt += '\n'


        # print(Task_Prompt)
        return Task_Prompt

    def answer_fusion(model, test_data,caption_dict,syn_question_dict,syn_answer_dict,ans_to_cap_dicts, config, answer_cap, rationale_cap, answer_long, rationale_long, answer_short, rationale_short):
        results = []
        answer_index = 0
        test_data = test_data[:2]
        for n, per_test_data in enumerate(test_data):

            cur_answer_cap = answer_cap[answer_index]['answer']
            cur_answer_long = answer_long[answer_index]['answer']
            cur_answer_short = answer_short[answer_index]['answer']
            answer_index += 1
            # print(per_test_data)
            question = per_test_data['question'].lower().strip()
            question_id = per_test_data['question_id']
            cur_rationale_cap = rationale_cap[str(question_id)]
            cur_rationale_long = rationale_long[str(question_id)]
            cur_rationale_long = AnswerFusion.get_string_before_first_period(cur_rationale_long)
            cur_rationale_short = rationale_short[str(question_id)]

            question_id_next = test_data[(n + 1) % len(test_data)]['question_id']
            ans_dict_queid = ans_to_cap_dicts[question_id]
            syn_question_queid = syn_question_dict[question_id]
            syn_question_queid_next = syn_question_dict[question_id_next]
            caption = caption_dict[question_id]
            syn_ans_queid = syn_answer_dict[question_id]
            Task_Prompt = AnswerFusion.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)

            Prompt = "Please answer the question with only one word based on the most reasonable rationale.\n"
            Prompt += f"Rationale: 1.The context mentions \"{cur_rationale_cap}\".\n" \
                    f"2.The background knowledge mentions \"{cur_rationale_long}\".\n" \
                    f"3.The background knowledge mentions \"{cur_rationale_short}\".\n"
            Prompt += Task_Prompt + "Question:" + question
            Prompt += f" {cur_answer_cap} or {cur_answer_long} or {cur_answer_short}?\n"
            Prompt += "Answer:"

            # Prompt for answering without rationales
            # Prompt = f"Please choose the correct answer from the three answer candidates.\n"
            # Prompt += Task_Prompt + "Question:" + question
            # Prompt += f" {cur_answer_cap} or {cur_answer_know0} or {cur_answer_know1}?\n"
            # Prompt += "Answer:"
            
            messages = [{'role':'user', 'content':Prompt}]
            pred_answer = llm_generate(model=model, messages=messages, temperature=0, max_tokens=1000)
            pred_answer = pred_answer['choices'][0]['message']['content']
            pred_answer = pred_answer.lower()
            pred_answer = post_process(pred_answer)
            print(f"ques_id: {question_id}  question: {question}  final_answer: {pred_answer} gt_answer: {per_test_data['answer']}\n")
            results.append({"question_id": question_id, "answer": pred_answer})
            for gtAns in per_test_data['answer']:
                if gtAns == pred_answer:
                    print("pass")
                    break
            print("fail")
        return results
    
    def answer_fusion_single(llm_client, question, answer_cap, rationale_cap, answer_long, rationale_long, answer_short, rationale_short, ans_dict_queid, syn_question_queid, caption, syn_ans_queid, config):
        cur_answer_cap = answer_cap
        cur_answer_long = answer_long
        cur_answer_short = answer_short
        # answer_index += 1
        # print(per_test_data)
        question
        # question_id = per_test_data['question_id']
        cur_rationale_cap = rationale_cap
        cur_rationale_long = rationale_long
        cur_rationale_long = AnswerFusion.get_string_before_first_period(cur_rationale_long)
        cur_rationale_short = rationale_short

        # question_id_next = test_data[(n + 1) % len(test_data)]['question_id'] # not needed (this uses a different question as a example)
        ans_dict_queid # generated by map_words_to_captions
        syn_question_queid # generate by generate_qa_pairs
        syn_question_queid_next = None # not needed (this uses a different question as a example)
        
        syn_ans_queid # generated by generate_qa_pairs
        Task_Prompt = AnswerFusion.create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)

        Prompt = "Please answer the question with only one word based on the most reasonable rationale.\n"
        Prompt += f"Rationale: 1.The context mentions \"{cur_rationale_cap}\".\n" \
                f"2.The background knowledge mentions \"{cur_rationale_long}\".\n" \
                f"3.The background knowledge mentions \"{cur_rationale_short}\".\n"
        Prompt += Task_Prompt + "Question:" + question
        Prompt += f" {cur_answer_cap} or {cur_answer_long} or {cur_answer_short}?\n"
        Prompt += "Answer:"

        # Prompt for answering without rationales
        # Prompt = f"Please choose the correct answer from the three answer candidates.\n"
        # Prompt += Task_Prompt + "Question:" + question
        # Prompt += f" {cur_answer_cap} or {cur_answer_know0} or {cur_answer_know1}?\n"
        # Prompt += "Answer:"
        
        messages = [{'role':'user', 'content':Prompt}]
        pred_answer = llm_generate(client=llm_client, model=config["llm_model"], messages=messages, temperature=0, max_tokens=1000)
        pred_answer = pred_answer.choices[0].message.content
        pred_answer = pred_answer.lower()
        pred_answer = post_process(pred_answer)
        print(f"question: {question}  final_answer: {pred_answer}\n")
        return pred_answer

def diet_coke_e2e(llm_client, caption_client, image, question, config):
    captions = caption_gen.generate_multiple_captions_vllm_multi(caption_client, config["vlm_model"], image, config['num_caps_per_img'])
    qa_pairs = qa_gen.generate_qa_pairs_vllm(llm_client, config["llm_model"], captions, sampling_params=None)
    ans_dict = word_to_caption.map_words_to_captions(captions)
    ans_dict_queid = word_to_caption.map_words_to_captions(captions)

    # generate question array and answer array
    syn_question = []
    syn_answer = []
    for qa_pair in qa_pairs:
        syn_question.append(qa_pair[0])
        syn_answer.append(qa_pair[1]+".")

    long_knowledge, short_knowledge = Knowledge.knowledge_generate_single(llm_client, question, ans_dict_queid, syn_question, captions, syn_answer, config)#ans_dict, syn_answer, captions, syn_question, config)
    pred_answer_long, pred_answer_short, pred_answer_cap = AnswerCandidates.answer_candidates_generate_single(llm_client, question, ans_dict_queid, syn_question, captions, syn_answer, config, long_knowledge, short_knowledge)
    rationale_long, rationale_short, rationale_cap = AutoRationales.auto_rationale_generate_single(llm_client, question, pred_answer_long, pred_answer_short, pred_answer_cap, ans_dict_queid, syn_question, captions, syn_answer, config)
    final_answer = AnswerFusion.answer_fusion_single(llm_client, question, pred_answer_cap, rationale_cap, pred_answer_long, rationale_long, pred_answer_short, rationale_short, ans_dict_queid, syn_question, captions, syn_answer, config)
    return final_answer

def main(args, config):

    #### Dataset ####
    print("Creating vqa datasets")
    test_data = []
    for f in config['test_file']:
        test_data = json.load(open(f, 'r'))


    caption_data = json.load(open(config['caption_file'], 'r'))
    quesID_to_cap_dict = create_cap_dic(caption_data)

    question_data = json.load(open(config['question_file'], 'r'))
    quesID_to_ques_data,syn_answer_dict = create_generated_question_dic(question_data)

    ans_dict_data = json.load(open(config['ans_dict_file'], 'r'))
    ans_to_cap_dicts = create_ans_to_cap_dic(ans_dict_data)

    result_filename = config['result_tag']+'_'+config['dataset']+'_'+config['model']+'_'+config['dist_selection'] + 'caps'+str(config['num_caps_per_img']) +'_question'+ str(config['num_question_per_img'])+'_questiontype'+'_'+config['question_type']

    print('knowledge generation')
    model = args.model
    if not args.evaluate_direct:
        knowledge_long, knowledge_short = Knowledge.knowledge_generate(model, test_data, quesID_to_cap_dict,quesID_to_ques_data,syn_answer_dict,ans_to_cap_dicts, config)
        print('save results')
        with open(config['output_dir'] + "long_knowledge.json", 'w') as json_file:
            json.dump(knowledge_long, json_file)
        with open(config['output_dir'] + "short_knowledge.json", 'w') as json_file:
            json.dump(knowledge_short, json_file)
    else:
        # load the available result file directly and run evaluation
        result_file = os.path.join(args.result_dir, '%s.json'%result_filename)
        print('Evaluate directly using ', result_file)

    print('answer candidate generation')
    if not args.evaluate_direct:
        answer_long, answer_short, answer_cap = AnswerCandidates.answer_candidates_generate(model, test_data, quesID_to_cap_dict,quesID_to_ques_data,syn_answer_dict,ans_to_cap_dicts, config, knowledge_long, knowledge_short)        
        # print('save results')
        # result_file = save_result(answer_long, args.result_dir, result_filename, remove_duplicate='question_id')
    else:
        # load the available result file directly and run evaluation
        result_file = os.path.join(args.result_dir, '%s.json'%result_filename)
        print('Evaluate directly using ', result_file)

    print("auto rationales generation")
    if not args.evaluate_direct:
        auto_rationale_long, auto_rationale_short, auto_rationale_cap = AutoRationales.auto_rationale_generate(model, test_data, quesID_to_cap_dict,quesID_to_ques_data,syn_answer_dict,ans_to_cap_dicts, config, answer_long, answer_short, answer_cap)
        print('save results')
        with open(config['output_dir'] + "rationale_long.json", 'w') as json_file:
            json.dump(auto_rationale_long, json_file)
    else:
        # load the available result file directly and run evaluation
        result_file = os.path.join(args.result_dir, '%s.json'%result_filename)
        print('Evaluate directly using ', result_file)

    print("answer fusion")
    if not args.evaluate_direct:
        vqa_result = AnswerFusion.answer_fusion(model, test_data, quesID_to_cap_dict,quesID_to_ques_data,syn_answer_dict,ans_to_cap_dicts, config, answer_cap, auto_rationale_cap, answer_long, auto_rationale_long, answer_short, auto_rationale_short)
        print('save results')
        # result_file = save_result(vqa_result, args.result_dir, result_filename, remove_duplicate='question_id')
    else:
        # load the available result file directly and run evaluation
        result_file = os.path.join(args.result_dir, '%s.json'%result_filename)
        print('Evaluate directly using ', result_file)


def update(params, args):

    params['model'] = args.model
    params['llm_model'] = args.llm_model
    params['vlm_model'] = args.vlm_model
    params['dist_selection'] = args.dist_selection

    params['dataset'] = args.dataset
    params['split_seed'] = args.split_seed
    params['num_sample'] = args.num_sample
    params['output_dir'] = args.output_dir
    params['test_server'] = args.test_server

    params['num_caps_per_img'] = args.num_caps_per_img
    params['num_question_per_img'] = args.num_question_per_img
    params['caption_file'] = args.caption_file

    params['question_file'] = args.question_file
    params['question_ppl_file'] = args.question_ppl_file
    params['ans_dict_file'] = args.ans_dict_file

    params['question_type'] = args.question_type

    params['random_question'] = args.random_question
    params['result_tag'] = args.result_tag
    params['evaluate_direct'] = args.evaluate_direct
    params['resume'] = args.resume

    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/AOKVQA_caption.yaml')
    parser.add_argument('--caption_file', default='../caption_question_files/aokvqa_val_caption.json')
    parser.add_argument('--question_file', default='../caption_question_files/aokvqa_val_question.json')
    parser.add_argument('--question_ppl_file', default=None)
    parser.add_argument('--ans_dict_file', default='../caption_question_files/aokvqa_val_ans_to_cap_dict.json')
    parser.add_argument('--question_type', default='g_q', type=str)

    parser.add_argument('--output_dir', default='./output_knowledge_aok/mistral_inst/')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument('--evaluate_direct', action='store_true')

    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--vqa_eval', action='store_true')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--split_seed', default=0, type=int)
    parser.add_argument('--num_sample', default=16, type=int)
    parser.add_argument('--ensemble', default=1, type=int)
    parser.add_argument('--random_question', action='store_true')
    parser.add_argument('--test_server', action='store_true')

    parser.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--llm_model', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--vlm_model', default='openai/clip-vit-base-patch16', type=str)
    parser.add_argument('--dist_selection', default='hugging', type=str)
    parser.add_argument('--select_cap', action='store_true')

    parser.add_argument('--dataset', default='aokvqa', type=str)
    parser.add_argument('--result_tag', default='', type=str)

    parser.add_argument('--batch_size_test', default=1, type=int)


    parser.add_argument('--num_caps_per_img', default=30, type=int)
    parser.add_argument('--num_question_per_img', default=30, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = update(config, args)
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)