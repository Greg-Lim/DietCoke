import en_core_web_sm
import random
from concurrent import futures

from .utils import *

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