#####################################
## This project includes code from:
## Copyright (c) 2025 Zhiwei He
## https://github.com/zwhe99/DeepMath
#####################################

import pandas as pd
import string
import json
import re
import math
from typing import Optional, Tuple,List
from openai import OpenAI
import yaml
import os
from dotenv import load_dotenv
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import json
import ray

##################################
from pm.miner import Miner
from pm.checker import Checker
import reward
import pm

truelogs=pd.read_csv('eventlogs/DeepMath_eventlog.csv')

def load_template(yaml_file='./reward/prompt.yaml'):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        template_data = yaml.safe_load(file)
    return template_data['answer_prompt']

def process_reward_func(think:str, index:int):
    think_log = reward.Answer2EventAgent().make_event_log(think)
    think_log.log['Case ID'] = str(index)
    reason_net = Miner(think_log).discover()
    
    true_log_df = truelogs[truelogs['Case ID'] == index].copy()
    true_log_df['Case ID'] = str(index)
    true_eventlog = pm.EventLog(true_log_df)

    conf_df = Checker(true_eventlog, reason_net).check()

    return conf_df


_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def _token_f1(a: str, b: str) -> float:
    a_tok = _normalize_text(a).split()
    b_tok = _normalize_text(b).split()
    if not a_tok and not b_tok:
        return 1.0
    if not a_tok or not b_tok:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a_tok), Counter(b_tok)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(ca.values()))
    recall = overlap / max(1, sum(cb.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_numbers(s: str) -> List[float]:
    import re

    s = s or ""
    nums = []
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            continue
    return nums


def answer_similarity_score(pred: str, gold: str) -> float:
    # number-aware quick match
    pnums, gnums = _extract_numbers(pred), _extract_numbers(gold)
    if pnums and gnums:
        for pn in pnums:
            for gn in gnums:
                if math.isfinite(pn) and math.isfinite(gn):
                    if abs(pn - gn) <= 1e-6 * max(1.0, abs(gn)):
                        return 1.0
    # fallback to token F1
    return _token_f1(pred, gold)



def split_solution_and_index(text):
    match = re.search(r'<index>(.*?)</index>', text)
    if match:
        index = match.group(1)
        solution = re.sub(r'<index>.*?</index>', '', text).strip()
    else:
        index = None
        solution = text.strip()
    return solution, index

def split_think_and_answer(text):
    parts = re.split(r'</think>', text, maxsplit=1, flags=re.DOTALL)

    if len(parts) == 2:
        think_part = parts[0]
        answer = parts[1].strip()

        # Remove <think> tag
        think = re.sub(r'^.*?<think>', '', think_part, flags=re.DOTALL).strip()

        return think, answer
    else:
        return text.strip(), "WRONG ANSWER"

def ensure_ray_initialized():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def accuracy_reward(completions, solution: list[str], **kwargs):
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(contents, solution):
        sol, _ = split_solution_and_index(sol[0])
        _,content=split_think_and_answer(content)
        
        try:
            gold_parsed = parse(sol)
        except Exception:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                reward = 0.0
        else:
            # fallback to text match
            reward = float(content.strip().lower() == sol.strip().lower())
        rewards.append(reward)

    return rewards

def think_format_reward(completions, solution: list[str], **kwargs):
    rewards = []
    contents = [completion[0]["content"] for completion in completions]
    for content in contents:
        if "</think>" in content:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def process_reward(completions, solution: list[str], **kwargs):
    # Ray-parallel version of the per-sample computation
    contents = [completion[0]["content"] for completion in completions]
    @ray.remote
    def _compute_conf_score(content: str, sol_text: str) -> float:
        try:
            # extract clean solution and index
            _, index = split_solution_and_index(sol_text[0])
            think, _ = split_think_and_answer(content)
            
            if index is None:
                print("Missing index in solution metadata; skipping process reward.")
                return 0.0
            process_reward_value = float(process_reward_func(think, int(index))['F1 Score'].values[0])
            print("PROCESS REWARD: ", process_reward_value)
            return process_reward_value
        except Exception as e:
            print(f"Error processing content: {e}")
            return 0.0

    ensure_ray_initialized()

    # Launch tasks in parallel and gather results
    futures = [_compute_conf_score.remote(content, sol) for content, sol in zip(contents, solution)]
    rewards = ray.get(futures)
    
    return rewards
