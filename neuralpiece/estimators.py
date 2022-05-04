import math

class UniformEstimator:

    def __init__(self, vocab_size: int) -> None:
        self.logprob = -math.log(vocab_size)

    def __call__(self, *whatever, **kwhatever) -> float:
        return self.logprob
