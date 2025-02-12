
from .knowledge_generate import Knowledge
from .auto_rationale_generate import AutoRationales
from .candidate_generate import AnswerCandidates
from .answer_fusion import AnswerFusion
import time

def diet_coke_e2e_lavis_serve(llm_client, get_lavis_smaples, image, question, config):
    timings = {}
    tik = time.time()
    lavis_samples = get_lavis_smaples(image, question)
    timings["lavis"] = time.time() - tik

    print(lavis_samples['captions'][0])
    print(lavis_samples['questions'])
    print(lavis_samples['answers'])
    print(lavis_samples['ans_to_cap_dict'])

    print("timings: ", timings)
    return "nothing"

    print("After LAVIS samples")

    ans_dict_queid = lavis_samples['ans_to_cap_dict']
    syn_question = lavis_samples['questions']
    syn_answer = lavis_samples['answers']
    captions = lavis_samples['captions'][0]
    
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