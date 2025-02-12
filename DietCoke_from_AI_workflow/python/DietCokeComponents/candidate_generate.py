import en_core_web_sm
import random
from concurrent import futures

nlp = en_core_web_sm.load()

from .utils import *

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
        