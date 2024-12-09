import argparse
import os
# os.environ['TRANSFORMERS_CACHE'] = '../'
import ruamel.yaml as yaml
import numpy as np
import random
import time
import gc
import copy
from collections import OrderedDict
import datetime
import json
import csv
from pathlib import Path
import pprint
from nltk.stem import WordNetLemmatizer

# assert timm.__version__ == '0.4.9'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from itertools import chain

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import snapshot_download
# from transformers import T5TokenizerFast
from huggingface_hub import login

import en_core_web_sm
nlp = en_core_web_sm.load()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


from .vqaTools.vqa import VQA

from . import utils
from .dataset.utils import vqa_eval, save_result
#from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

import nltk
from nltk.corpus import wordnet


def load_model(model_selection,dist_selection):
    if (model_selection[:3] == 'opt' and dist_selection == 'hugging'):
        checkpoint = 'facebook/'+model_selection
        if model_selection == 'opt-175b':
            cache_dir = '/export/share/jiaxian-guo/OPT_175B'
        else:
            cache_dir = '../models/'+model_selection

        from huggingface_hub import hf_hub_download
        print(hf_hub_download(repo_id=checkpoint, filename="config.json"))

        if model_selection != 'opt-175b':
            print("load..{}".format(checkpoint))
            weights_path = snapshot_download(repo_id=checkpoint,
                                             ignore_patterns=["*.msgpack", "*.h5", "*msgpack.index.json",
                                                              "*h5.index.json"])
                                             # cache_dir=cache_dir)  # avoid downloading the useless files

            # weights_path = snapshot_download(repo_id=checkpoint,ignore_patterns=["*.msgpack","*.h5","*msgpack.index.json","*h5.index.json"],cache_dir=cache_dir) # avoid downloading the useless files
        else: # the weights of OPT_175B are self-processed
            weights_path = cache_dir
        import os
        files = os.listdir(weights_path)

        print(files)
        # weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path
        config_opt = AutoConfig.from_pretrained(weights_path)

        # Initializes an empty shell with the model. This is instant and does not take any RAM.
        with init_empty_weights():
            # model = AutoModelForCausalLM.from_pretrained(cache_dir)
            model = AutoModelForCausalLM.from_config(config_opt)
        # Initialize the model under the previous context manager breaks the tied weights.
        model.tie_weights()
        # Infer device map automatically
        if model_selection == 'opt-66b':   # the weight map of 'opt_66B is different from others
            device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')
        else:
            device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')
        print(device_map)

        num_decoder_layer = len(device_map)


        # Because the input may occupy many GPU memory in the first GPU, so we need to reduce the number of  model layers in the first GPU.
        if model_selection == 'opt-66b':
            num_gpu = 4
            num_layer_first_gpu = 16
        elif model_selection == 'opt-30b':
            num_gpu = 2
            num_layer_first_gpu = 24
        elif model_selection == 'opt-175b':
            num_gpu = 16
            num_layer_first_gpu = 2
        if model_selection not in ['opt-125m','opt-13b','opt-2.7b','opt-6.7b']:
            num_layer_per_gpu = (num_decoder_layer - num_layer_first_gpu) / (num_gpu-1)
            for i, layer in enumerate(device_map):
                if i > (num_layer_first_gpu):
                    device_map[layer] = int((i - num_layer_first_gpu) / num_layer_per_gpu) + 1
            print(device_map)

        if any([k == 'disk' for k in device_map.values()]):
            offload_folder = 'offload_folder'
        else:
            offload_folder = None
        if model_selection not in ['opt-125m', 'opt-2.7b']:
            if model_selection == 'opt-66b':  # the weight map of 'opt_66B is different from others
                load_checkpoint_and_dispatch(
                    model,
                    weights_path,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    dtype='float16',
                    offload_state_dict=True
                )
            else:
                load_checkpoint_and_dispatch(
                    model.model,
                    weights_path,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    dtype='float16',
                    offload_state_dict=True
                )
            model.tie_weights()

            if model_selection == 'opt-66b':
                full_model_device_map = {f"{k}": v for k, v in device_map.items()}
            else:
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
            full_model_device_map["lm_head"] = 0
            dispatch_model(model, device_map=full_model_device_map)
        if model_selection == 'opt-175b':
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
            # tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    if (args.model_selection[:3] == 'opt') and args.dist_selection == 'alpa':
        # from transformers import AutoTokenizer
        from opt_serving.model.wrapper import get_model
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        tokenizer.add_bos_token = False

        model_name = 'alpa/'+args.model_selection.split('_')[0]+'-'+args.model_selection.split('_')[1].lower()
        model_path = '/export/home/alpa_checkpoint/'+args.model_selection.split('_')[1]+"/numpy_checkpoint/"
        # Load the model
        model = get_model(model_name=model_name,
                          device="cpu",
                          encoder_seq_lengths=[1,256,1024],
                          batch_size=args.ensemble,
                          # autoregressive=False,
                          path=model_path)

    if model_selection == 'mistral-7b':
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    return model,tokenizer

def create_cap_dic(caption_data):
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
    # words_list = pred_answer.split()
    # words_list_new = [lemmatize_word(w) for w in words_list]
    # pred_answer = " ".join(words_list_new)
    return pred_answer

def get_string_before_first_period(text):
    period_index = text.find('.')
    if period_index != -1:
        return text[:period_index]
    return text

@torch.no_grad()
def evaluation(model, test_data,caption_dict,syn_question_dict,syn_answer_dict,ans_to_cap_dicts,tokenizer, device, config, logger, writer, split='test',resume=False,result_dir = 'output/VQA_caption/result', result_filename = 'result'):
    # test
    if  config['dist_selection']  != 'alpa':
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    header = 'Generate VQA test result:'
    print_freq = 50
    if resume:
        resume_result_file = os.path.join(result_dir, '%s.json'%result_filename)
        result = json.load(open(resume_result_file, 'r'))
    else:
        result = []
    tested_quesId_dict = {}
    # print(result)
    for tested_dict in result:
        # print(result)
        if tested_dict['answer'] is not None:
            tested_quesId_dict[tested_dict['question_id']] = 1
    # print(test_data)

    answer_cap = json.load(open(config['answer_cap'], 'r'))
    rationale_cap = json.load(open(config['rationale_cap'], 'r'))
    rationale_cap_g = json.load(open(config['rationale_cap_g'], 'r'))
    answer_know0 = json.load(open(config['answer_know0'], 'r'))
    rationale_know0 = json.load(open(config['rationale_know0'], 'r'))
    rationale_know0_g = json.load(open(config['rationale_know0_g'], 'r'))
    answer_know1 = json.load(open(config['answer_know1'], 'r'))
    rationale_know1 = json.load(open(config['rationale_know1'], 'r'))
    rationale_know1_g = json.load(open(config['rationale_know1_g'], 'r'))
    answer_index = 0
    for n, per_test_data in enumerate(metric_logger.log_every(test_data, print_freq, header)):

        cur_answer_cap = answer_cap[answer_index]['answer']
        cur_answer_know0 = answer_know0[answer_index]['answer']
        cur_answer_know1 = answer_know1[answer_index]['answer']
        answer_index += 1
        # print(per_test_data)
        question = per_test_data['question'].lower().strip()
        question_id = per_test_data['question_id']
        cur_rationale_cap = rationale_cap[str(question_id)]
        cur_rationale_cap_g = rationale_cap_g[str(question_id)]
        cur_rationale_know0 = rationale_know0[str(question_id)]
        cur_rationale_know0 = get_string_before_first_period(cur_rationale_know0)
        cur_rationale_know0_g = rationale_know0_g[str(question_id)]
        cur_rationale_know1 = rationale_know1[str(question_id)]
        cur_rationale_know1_g = rationale_know1_g[str(question_id)]

        if resume and question_id in tested_quesId_dict.keys():
            continue
        question_id_next = test_data[(n + 1) % len(test_data)]['question_id']
        ans_dict_queid = ans_to_cap_dicts[question_id]
        syn_question_queid = syn_question_dict[question_id]
        syn_question_queid_next = syn_question_dict[question_id_next]
        caption = caption_dict[question_id]
        syn_ans_queid = syn_answer_dict[question_id]
        Task_Prompt = create_task_prompt(syn_question_queid, syn_ans_queid, syn_question_queid_next, config)

        Prompt = "Please answer the question with only one word based on the most reasonable rationale.\n"
        Prompt += f"Rationale: 1.{cur_rationale_cap_g} The context mentions \"{cur_rationale_cap}\".\n" \
                  f"2.{cur_rationale_know0_g} The background knowledge mentions \"{cur_rationale_know0}\".\n" \
                  f"3.{cur_rationale_know1_g} The background knowledge mentions \"{cur_rationale_know1}\".\n"
        Prompt += Task_Prompt + "Question:" + question
        Prompt += f" {cur_answer_cap} or {cur_answer_know0} or {cur_answer_know1}?\n"
        Prompt += "Answer:"

        # Prompt for answering without rationales
        # Prompt = f"Please choose the correct answer from the three answer candidates.\n"
        # Prompt += Task_Prompt + "Question:" + question
        # Prompt += f" {cur_answer_cap} or {cur_answer_know0} or {cur_answer_know1}?\n"
        # Prompt += "Answer:"
        
        encodeds = tokenizer(Prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generate_ids = model.generate(model_inputs.input_ids,
                                      max_length=config['max_answer_length'] + len(model_inputs.input_ids[0]))
        pred_answer = tokenizer.batch_decode(generate_ids[:, len(model_inputs.input_ids[0]):])[0]
        pred_answer = pred_answer.lower()
        pred_answer = post_process(pred_answer)

        result.append({"question_id": question_id, "answer": pred_answer})
        if n % 1000 == 1:
            save_result(result, result_dir, result_filename, remove_duplicate='question_id')
    return result



def main(args, config, logger, writer):
    # utils.init_distributed_mode(args)

    device = torch.device(args.device)
    if args.dist_selection == 'alpa':
        device = 'cpu'
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # breakpoint()


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


    if config['val_ann_path']:
        vqa = VQA(config['val_ann_path'], config['val_ques_path'])


    result_filename = config['result_tag']+'_'+config['dataset']+'_'+config['model_selection']+'_'+config['dist_selection'] + 'caps'+str(config['num_caps_per_img']) +'_question'+ str(config['num_question_per_img'])+'_questiontype'+'_'+config['question_type']
    start_time = time.time()

    print('evaluating')
    if not args.evaluate_direct:
        model, tokenizer = load_model(model_selection=args.model_selection, dist_selection=args.dist_selection)
        vqa_result = evaluation(model, test_data, quesID_to_cap_dict,quesID_to_ques_data,syn_answer_dict,ans_to_cap_dicts,tokenizer, device, config, logger=logger, writer=writer,resume=args.resume,result_dir = args.result_dir, result_filename = result_filename)
        print('save results')
        result_file = save_result(vqa_result, args.result_dir, result_filename, remove_duplicate='question_id')
    else:
        # load the available result file directly and run evaluation
        result_file = os.path.join(args.result_dir, '%s.json'%result_filename)
        print('Evaluate directly using ', result_file)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        rsl_keys = ['filename', 'overall_acc','other', 'yes/no', 'number', 'open', 'binary', 'comment']
        rsl_val = ['NA' for _ in rsl_keys]
        rsl = dict(zip(rsl_keys, rsl_val))
        rsl['filename'] = args.output_dir.split('/')[-1]

        rsl_csv_filename = config['result_tag']+'_'+config['dataset']+'_'+config['model_selection']+'_'+config['dist_selection'] + 'caps'+str(config['num_caps_per_img']) +'_question'+ str(config['num_question_per_img'])+'_questiontype'+'_'+config['question_type']


        rsl_csv_file = os.path.join(args.result_dir, '%s.csv' % rsl_csv_filename)
        if not args.test_server:
            vqaEval = vqa_eval(vqa, result_file, config['val_ques_path'], logger=logger, dataset=config['dataset'])
            rsl['overall_acc'] = vqaEval.accuracy['overall']
            if 'other' in vqaEval.accuracy['perAnswerType'].keys():
                rsl['other'] = vqaEval.accuracy['perAnswerType']['other']
            if 'yes/no' in vqaEval.accuracy['perAnswerType'].keys():
                rsl['yes/no'] = vqaEval.accuracy['perAnswerType']['yes/no']
            if 'number' in vqaEval.accuracy['perAnswerType'].keys():
                rsl['number'] = vqaEval.accuracy['perAnswerType']['number']
            else:
                rsl['overall_acc'] = vqaEval.accuracy['overall']
                for key in vqaEval.accuracy['perAnswerType'].keys():
                    rsl[key] = vqaEval.accuracy['perAnswerType'][key]
        print(rsl_csv_file)
        with open(rsl_csv_file, 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=rsl_keys)
            csv_writer.writerow(rsl)
        print('Done saving to csv file')
    writer.close()


def update(params, args):


    params['min_answer_length'] = args.min_answer_length
    params['max_answer_length'] = args.max_answer_length
    params['model_selection'] = args.model_selection
    params['dist_selection'] = args.dist_selection

    params['dataset'] = args.dataset
    params['split_seed'] = args.split_seed
    params['num_sample'] = args.num_sample
    params['output_dir'] = args.output_dir
    params['test_server'] = args.test_server
    params['answer_cap'] = args.answer_cap
    params['rationale_cap'] = args.rationale_cap
    params['rationale_cap_g'] = args.rationale_cap_g
    params['answer_know0'] = args.answer_know0
    params['rationale_know0'] = args.rationale_know0
    params['rationale_know0_g'] = args.rationale_know0_g
    params['answer_know1'] = args.answer_know1
    params['rationale_know1'] = args.rationale_know1
    params['rationale_know1_g'] = args.rationale_know1_g

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
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" #this work together with dist barrier timeout
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/AOKVQA_caption.yaml')
    parser.add_argument('--caption_file', default='../caption_question_files/aokvqa_val_caption.json')
    parser.add_argument('--question_file', default='../caption_question_files/aokvqa_val_question.json')
    parser.add_argument('--question_ppl_file', default=None)
    parser.add_argument('--answer_cap', default='') # candidate_cap
    parser.add_argument('--rationale_cap', default='')  # MR_cap
    parser.add_argument('--rationale_cap_g', default='') # AR_cap
    parser.add_argument('--answer_know0', default='') # candidate_short
    parser.add_argument('--rationale_know0', default='') # MR_short
    parser.add_argument('--rationale_know0_g', default='') # AR_short
    parser.add_argument('--answer_know1', default='') # candidate_long
    parser.add_argument('--rationale_know1', default='') # MR_long
    parser.add_argument('--rationale_know1_g', default='') # AR_long

    parser.add_argument('--ans_dict_file', default='../caption_question_files/aokvqa_val_ans_to_cap_dict.json')
    parser.add_argument('--question_type', default='g_q', type=str)

    parser.add_argument('--output_dir', default='')
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



    parser.add_argument('--model_selection', default='mistral-7b', type=str)
    parser.add_argument('--dist_selection', default='hugging', type=str)
    parser.add_argument('--select_cap', action='store_true')

    parser.add_argument('--dataset', default='aokvqa', type=str)
    parser.add_argument('--result_tag', default='', type=str)

    parser.add_argument('--batch_size_test', default=1, type=int)


    parser.add_argument('--num_caps_per_img', default=30, type=int)
    parser.add_argument('--num_question_per_img', default=30, type=int)

    parser.add_argument('--min_answer_length', default=1, type=int,
                        help='min answer length during inference (generate); '
                             'None  == self.model.config.min_length (0 for t0)')
    parser.add_argument('--max_answer_length', default=10, type=int,
                        help='max answer length during inference (generate); '
                             'None  == self.model.config.max_length (20 for t0)')
    args = parser.parse_args()


    assert args.model_selection in ['opt-30b','opt-66b','opt-175b','opt-13b','opt-6.7b', 'mistral-7b', 'gemma-7b']
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = update(config, args)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    logger, writer = utils.setup_default_logging(args)
    # args.rsl_csv = os.path.join('./csv_folder', args.rsl_csv)
    main(args, config, logger, writer)
