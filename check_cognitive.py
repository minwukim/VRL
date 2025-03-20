import json
import pandas as pd
import time
import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from datasets import load_dataset

# bfloat16 not supported by v100 gpus
model = "/home/VRL/outputs/qwen2.5-3b-grpo-large/checkpoint-300"
# model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# llm = LLM(model="Qwen/Qwen2.5-3B-Instruct")
llm = LLM(model=model, max_model_len=3000)

params = SamplingParams(
    n=8,
    temperature=0.9,
    max_tokens=2048
)

SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

test = load_dataset("HuggingFaceH4/MATH-500", split="test")
test = test.map(lambda x: {
        "prompt": SYSTEM.format(prompt=x["problem"]),
        "answer": x["answer"],
        "level": x["level"]
        })
test = test.remove_columns(["problem", "solution", "subject", "unique_id"])

outputs = llm.generate(test['prompt'], params)

# Use regex to extract only the {prompt} content
pattern = r'by the \\boxed{} tag.\s*(.*?)\s*Assistant:'
match = re.search(pattern, SYSTEM, re.DOTALL)
inferences = {'answers': []}
for out in (outputs):
    question = re.search(pattern, out.prompt, re.DOTALL).group(1).strip()
#     print(question)
    for ans in out.outputs:
        inferences['answers'].append(ans.text)

df = pd.DataFrame(inferences)

df['rethink'] = df['answers'].str.contains("rethink").astype(int)
df['recheck'] = df['answers'].str.contains("recheck").astype(int)
df['reevaluate'] = df['answers'].str.contains("re-evaluate").astype(int)
df['try'] = df['answers'].str.contains("try again").astype(int)
df['check_again'] = (df['answers'].str.contains("check again").astype(int))
df['hmm'] = (df['answers'].str.contains("Hmm").astype(int))
df['verify'] = df['answers'].str.contains("verify").astype(int)
df['wait'] = df['answers'].str.contains("Wait").astype(int)

df.to_csv("qwen_cp300_cog.csv", index=False)
print(df.sum())


