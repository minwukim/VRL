from datasets import load_dataset
import re


def reward_kk(output, ground_truth):
    def preprocess(text):
       # pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
       # match = re.search(pattern, text, re.DOTALL)
       # if match: text = match.group(1)

        lines = text.split("\n")
        solution = {}
        
        for line in lines:
            sp = line.split(" ")
            if not line: continue
            if len(sp) < 2: continue

            solution[sp[1].lower()] = sp[-1].lower()
        return solution

         
    pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
    s = output 
    if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
        # incorrect amount of tokens
        return -2 
    match = re.search(pattern, s, re.DOTALL)
    # if answer doesn't match provided format
    if not match: return -2

    # answer format is correct now
    # 
    
    gt = preprocess(ground_truth)
    out = preprocess(match.group(1).strip())
    print(out)
    names = out.keys()
    for name in names:
        val = gt.get(name, None)
        if val is None: return -1

    roles = out.values()
    for role in roles:
        if role.lower() != 'knight' and role.lower() != 'knave': return -1

    # formatting of final answer also good
    return 2 if gt == out else -0.5
def get_kk_test_prompts():


    SYSTEM_PROMPT = """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
    i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: Please solve this logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>. Here is my question,
    
    {prompt}
    Assistant: <think>
    """
    ds = load_dataset("parquet", data_files="/home/VRL/vanilla_trainer/Logic-RL/3ppl/test.parquet")["train"]
    prompts = []
    ground_truths = []

    for example in ds:
       prompt = SYSTEM_PROMPT.format(prompt=example['quiz'])
       gt = example['solution_text_format']
       prompts.append(prompt)
       ground_truths.append(gt)

    return prompts, ground_truths, reward_kk

