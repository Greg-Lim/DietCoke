import en_core_web_sm
import random
from concurrent import futures

nlp = en_core_web_sm.load()

from .utils import *



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

