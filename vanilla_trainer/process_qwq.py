import pandas as pd
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")

def check_pattern(text):
    pattern = r".+</think>\s*<answer>(.+)</answer>\s*$"
    s =text 
    if not (s.count("</think>") == 1 and s.count("<answer>") == 1 and s.count("</answer>") == 1):
        # incorrect amount of tokens
        #print("counts no match")
        return -2 
    match = re.search(pattern, s, re.DOTALL)
    if not match: return -2
    return 0

df = pd.read_csv("qwq_samples_more.csv")
print(df.columns)

df['llm_answer'] = df['llm_answer'] + "</answer>"
df['pattern'] = [check_pattern(text) for text in df['llm_answer']]
df['tokens'] = [len(tokenizer(text)['input_ids']) for text in df['llm_answer']]


df.to_csv("processed_qwq_samples.csv", index=False)
