import random


def check(code: str) -> int:
    max_val = len(code)
    min_val = random.random()
    x = random.randint(a=1, b=11)
    return (x - min_val) / (max_val - min_val)
