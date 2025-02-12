

import en_core_web_sm
import nltk
from nltk.stem import WordNetLemmatizer

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