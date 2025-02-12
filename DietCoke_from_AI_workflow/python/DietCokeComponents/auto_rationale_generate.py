
import en_core_web_sm
import random
from concurrent import futures

nlp = en_core_web_sm.load()

from .utils import *

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