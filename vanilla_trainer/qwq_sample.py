import os
from vllm import LLM, SamplingParams
import pandas as pd
from datasets import load_dataset


from dataset_loader import load_math, load_countdown, load_kk


SYSTEM="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> final answer inside \\boxed{{}} tag </answer>. User: You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

train, test, _ = load_kk()

sampling_params = SamplingParams(temperature=0, max_tokens=3000)
llm = LLM(model="Qwen/QwQ-32B", max_model_len=3100, gpu_memory_utilization=0.8, tensor_parallel_size=2)

process_in = train['prompt']
outputs = llm.generate(process_in, sampling_params)
process_out = [output.outputs[0].text for output in outputs]

df = pd.DataFrame()
df['question'] = process_in
df['llm_answer'] = process_out

df.to_csv("qwq_samples.csv", index=False)


