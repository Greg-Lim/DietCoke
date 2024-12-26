from openai import OpenAI
import pytest
from qa_gen import generate_qa_pairs_vllm
from vllm import SamplingParams

@pytest.fixture(scope="module")
def client():
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="idk"
    )
    return client


@pytest.fixture(scope="module")
def captions():
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

    return captions

# pytest -v VL_captioning/QAPairGen/test_qa_gen.py -k test_vllm_generate_qa_pairs -s
def test_vllm_generate_qa_pairs(client, captions):
    model = "mistralai/Mistral-7B-Instruct-v0.2"
    qa_pairs = generate_qa_pairs_vllm(client, model, captions, sampling_params=SamplingParams(temperature=0.2))
    print("vllm QA Pairs:", qa_pairs)
    print("Total Unique:", len(set(qa_pairs)))
    print("Total Captions:", len(captions))
    assert len(qa_pairs) == len(captions)


