import os
import re
from datasets import load_dataset, concatenate_datasets

from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string


def extract_boxed_answer(solution):
    return last_boxed_only_string(solution)

def load_math(SYSTEM, sample=None, diff_filter=False):
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")

    #sample 'n' from train
    if sample is not None:
        train = train.train_test_split(train_size=100, shuffle=True, seed=42)['train']

    train = train.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": extract_boxed_answer(x["solution"]),
        "level": x["level"]
        })

    test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["answer"],
        "level": x["level"]
        })

    if diff_filter:
        train = train.filter(lambda x: (x['level'] == 'Level 4') or (x['level'] == 'Level 5'))
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test

def load_countdown(SYSTEM):
    pass

def load_kk():  
    def reward_kk(completions, answer, **kwargs):
        def calculate_reward(output, ground_truth):
            def preprocess(text):
               # pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
               # match = re.search(pattern, text, re.DOTALL)
               # if match: text = match.group(1)

                lines = text.split("\n")
                solution = {}
                
                for line in lines:
                    sp = line.strip().split(" ")
                    if not line: continue
                    if len(sp) < 2: continue

                    solution[sp[1].lower()] = sp[-1].lower()
                return solution 

            pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
            s = output 
            if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
                # incorrect amount of tokens
                #print("counts no match")
                return -2 
            match = re.search(pattern, s, re.DOTALL)
            # if answer doesn't match provided format
            if not match: return -2

            # answer format is correct now
            # 
            
            gt = preprocess(ground_truth)
            out = preprocess(match.group(1).strip())
            names = out.keys()
            for name in names:
                val = gt.get(name, None)
                if val is None: return -1

            roles = out.values()
            for role in roles:
                if 'knight' not in role.lower() and 'knave' not in role.lower(): return -1

            # formatting of final answer also good
            return 2 if gt == out else -0.5
        
        return [calculate_reward(c, gt) for c, gt in zip(completions, answer)]
    

    SYSTEM_PROMPT = """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
    i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: Please solve this logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>. Here is my question,
    
    {prompt}
    Assistant: <think>
    """
    data_path = "/home/VRL/vanilla_trainer/Logic-RL"
    
    ppl_ds = []
    for ppl_file in os.listdir(data_path):
        ds = load_dataset("parquet", data_files=f"{data_path}/{ppl_file}/train.parquet")["train"]
        ds = ds.add_column(name="file", column=[ppl_file]*len(ds))
        ppl_ds.append(ds)
    
    ppl_ds = concatenate_datasets(ppl_ds)
    

    ppl_ds_test = []
    for ppl_file in os.listdir(data_path):
        ds = load_dataset("parquet", data_files=f"{data_path}/{ppl_file}/test.parquet")["train"]
        ds = ds.add_column(name="file", column=[ppl_file]*len(ds))
        ppl_ds_test.append(ds)
    
    ppl_ds_test = concatenate_datasets(ppl_ds_test)


    ppl_ds = ppl_ds.map(lambda x: {
        "prompt": SYSTEM_PROMPT.format(prompt=x["quiz"]),
        "answer": x["solution_text_format"],
        })


    ppl_ds_test = ppl_ds_test.map(lambda x: {
        "prompt": SYSTEM_PROMPT.format(prompt=x["quiz"]),
        "answer": x["solution_text_format"],
        })

    return ppl_ds, ppl_ds_test, reward_kk

