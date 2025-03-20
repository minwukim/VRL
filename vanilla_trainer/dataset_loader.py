from datasets import load_dataset

from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string


def extract_boxed_answer(solution):
    return last_boxed_only_string(solution)

def load_math(SYSTEM):
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")

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
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test

def load_countdown(SYSTEM):
    pass
