
from .knowledge_generate import Knowledge
from .auto_rationale_generate import AutoRationales
from .candidate_generate import AnswerCandidates
from .answer_fusion import AnswerFusion
import time

def diet_coke_e2e(llm_client, question, ans_dict_queid, syn_question, syn_answer, captions, config):
    timings = {}
    
    tik = time.time()
    long_knowledge, short_knowledge = Knowledge.knowledge_generate_single(llm_client, question, ans_dict_queid, syn_question, captions, syn_answer, config)#ans_dict, syn_answer, captions, syn_question, config)
    timings["knowledge"] = time.time() - tik

    tik = time.time()
    pred_answer_long, pred_answer_short, pred_answer_cap = AnswerCandidates.answer_candidates_generate_single(llm_client, question, ans_dict_queid, syn_question, captions, syn_answer, config, long_knowledge, short_knowledge)
    timings["candidates"] = time.time() - tik

    tik = time.time()
    rationale_long, rationale_short, rationale_cap = AutoRationales.auto_rationale_generate_single(llm_client, question, pred_answer_long, pred_answer_short, pred_answer_cap, ans_dict_queid, syn_question, captions, syn_answer, config)
    timings["rationales"] = time.time() - tik

    tik = time.time()
    final_answer = AnswerFusion.answer_fusion_single(llm_client, question, pred_answer_cap, rationale_cap, pred_answer_long, rationale_long, pred_answer_short, rationale_short, ans_dict_queid, syn_question, captions, syn_answer, config)
    timings["fusion"] = time.time() - tik

    print(timings)
    
    return final_answer