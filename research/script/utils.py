from script.prompt_template import *
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from typing import Dict, Any, Union
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from tqdm import trange
import os
import random
from datetime import datetime
import numpy as np
import pickle as pkl
import json

def min_max_scaler(data):
    min_val = min(data)
    max_val = max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data


def save_pkl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pkl.dump(data, file)


def load_pkl(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data


def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def pkl2jsonl(data, file_name):
    with open(file_name, "w") as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + "\n")


def cls_n2t(ans):
    ans = str(ans)
    if "1" in ans:
        return "accept"
    elif "0" in ans:
        return "refuse"


def cls_t2n(ans):
    ans = str(ans)
    if "accept" in ans:
        return 1
    elif "refuse" in ans:
        return 0
    else:
        return ""


def label2text(label):
    if label == 1:
        return "accept."
    elif label == 0:
        return "refuse."


def click2bool(label):
    label = int(label)
    if label == 1:
        return True
    elif label == 0:
        return False


def rec_t2n(brand):
    brand = str(brand)
    return clean_brand(brand)


def clean_brand(text):
    text = ', '.join(
        text.replace("brand", "").replace("'", "").replace(":", "").replace('"','') .replace('{','').replace('}','').replace('.','').replace(" ", "").split(',')
    ) + "."
    return text

def brand2list(text):
    text = text.replace('\n','').replace('.','').replace(' ','').split(',')
    text = [i for i in text if i != '']
    return text


def convert_confidence(score):
    score = str(score)
    pattern = r"\d+"
    value = re.findall(pattern, score)
    if value != []:
        value = max(int(i) for i in value if int(i) <= 100)
    else:
        value = 0
    return value


def string2dict(data, task="cls"):
    # \'answer\': answer, \'confidence\': confidence, \'reason\': 'reason'
    ans_start = data.find("answer") if data.find("answer") != -1 else 0
    ans_start = (
        data.find("response")
        if data.find("response") != -1 and ans_start != 0
        else ans_start
    )
    conf_start = data.find("confidence")
    rea_start = data.find("reason")
    if task == "cls":
        answer = cls_t2n(data[ans_start:conf_start])
    elif task == "rec":
        answer = rec_t2n(data[ans_start:conf_start])

    confidence = convert_confidence(data[conf_start:rea_start])
    reason = data[rea_start:].replace("<|eot_id|>", "")
    data = {"answer": answer, "confidence": confidence, "reason": reason}
    return data


def get_amz_template(args, entry, template):

    n_shot_example = '\n'
    if args.task_name == 'rec':
        
        simi_prompt = """Reference: Below are category preferences from similar customer profiles. To effectively analyze these cases and predict brand preferences, follow these pattern analysis guidelines:

Pattern Analysis Guidelines:
1. Shopping Activity Patterns [Core Behavior Signals]
   - HIGH ACTIVITY: Frequent category purchases, high purchase-to-view ratio
   - MEDIUM ACTIVITY: Regular purchases in specific categories 
   - LOW ACTIVITY: Mainly browsing, low purchase conversion

2. Channel Preference Patterns [Purchase Behavior]
   - PURCHASE FOCUSED: High purchase rate in specific categories
   - BROWSE FOCUSED: High views but low purchase rate
   - MIXED PATTERN: Balanced purchase-browse ratio

3. Category Interest Signals [Shopping Focus]
   Strong Interest Indicators:
   - Repeated purchases in same category
   - Purchases across complementary categories
   - Recent purchase history in category
   Weak Interest Indicators:
   - Only category views without purchases
   - Random category browsing
   - Isolated one-time purchases

4. Engagement Level [Purchase Pattern]
   - HIGHLY ENGAGED: Regular category-specific purchases
   - MODERATELY ENGAGED: Occasional category purchases
   - MINIMALLY ENGAGED: Mainly browsing activity

Analysis Steps:
1. First identify actual purchase patterns
2. Look for complementary category purchases
3. Check purchase-to-view ratios in key categories
4. Validate with recent purchase history

Reference Cases (Please analyze step by step):\n"""

    else:
        simi_prompt = """Reference: Below are promotional responses from similar customer profiles. To effectively analyze these cases and predict future responses, follow these pattern analysis guidelines:

Pattern Analysis Guidelines:
1. Customer Engagement Patterns [Core Activity Signals]
   - HIGH ENGAGEMENT: Frequent visits, regular purchases, diverse categories
   - MEDIUM ENGAGEMENT: Moderate visit/purchase frequency, some category diversity
   - LOW ENGAGEMENT: Infrequent visits/purchases, limited category exploration

2. Purchase Value Patterns [Spending Behavior]
   - HIGH VALUE: Large total spend, high average transaction value
   - MEDIUM VALUE: Moderate total spend and transaction value
   - LOW VALUE: Small total spend, low average transaction value

3. Deal Receptiveness Signals [Critical Predictors]
   Positive Patterns:
   - Multiple deal opt-in history
   - Predominantly mobile usage
   - Recent and frequent purchase behavior
   Negative Patterns:
   - No previous deal participation
   - Predominantly desktop usage
   - Irregular purchase patterns

4. Points Engagement Level [Loyalty Indicators]
   - ACTIVE: Regular points usage, multiple deal participations
   - MODERATE: Occasional points usage, some deal participation
   - INACTIVE: Minimal points usage, rare deal participation

Analysis Steps:
1. First classify each case into these pattern groups
2. Compare relative metrics rather than absolute values
3. Note which patterns most strongly correlate with acceptance/refusal

Reference Cases (Please analyze step by step):\n"""

    if "causal" in args.ablation:
        if args.task_name == 'rec':
            causal_intro = """Reference: Factors and their importance ranking that affect brand recommendations based on historical data. To analyze these recommendation patterns, follow these guidelines:
Factor Analysis Guidelines:
    - Focus on proven purchase patterns
    - Consider complementary brand relationships
    - Look for consistent buying behavior
    - Value category purchase combinations
    
When reviewing the factors below, apply these principles to evaluate their predictive value:\n"""
        else:
            causal_intro = """Reference: Factors and their importance ranking that affect promotion effectiveness based on historical data. To analyze these promotional factors, follow these guidelines:

Factor Analysis Guidelines:
   - Identify primary vs supporting factors
   - Note which factors directly relate to deals/promotions
   - Consider which show customer's active choices
   - Check for relationships between factors
   
When reviewing the factors below, apply these principles to evaluate their predictive value:\n"""
    
    
    if "causal" in args.ablation:
        causal_text = '\n'.join(entry["causal_text"].split(','))
        n_shot_example = n_shot_example + causal_intro + causal_text + '\n'
        
    if "simi" in args.ablation:
        n_shot_example = n_shot_example + simi_prompt + entry["similar_text"] + '\n'    

    instruction = template["start"] + n_shot_example + template["end"]


    if args.strategy == 'human':
        conversation = entry['human_template']
    elif args.strategy == 'adapt':
        conversation = entry['adaptive_template']

    return instruction, conversation

