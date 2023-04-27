
from typing import List, Dict
import math

import numpy as np
from scipy.special import logsumexp


def viterbi_segment(
        word: str, vocab: Dict[str, float],
        inference_mode: str = "sum",
        is_bert_wordpiece: bool = False,
        sample: bool = False) -> List[str]:

    if inference_mode not in ["sum", "maxmin"]:
        return ValueError(
            "Inference mode must be either 'sum' or 'maxmin'.")

    if inference_mode == "sum":
        costs = [0. for _ in range(len(word) + 1)]
    elif inference_mode == "maxmin":
        costs = [1e9 for _ in range(len(word) + 1)]
    else:
        raise ValueError("Unkown inference mode.")
    prev = [0 for _ in word]

    # First, dynamic programming
    for i in range(1, len(word) + 1):
        scores = []
        indices = []
        for j in range(i):
            subword_candidate = word[j:i]
            if is_bert_wordpiece and j > 0:
                subword_candidate = "##" + subword_candidate
            if subword_candidate in vocab:
                if inference_mode == "sum":
                    new_cost = costs[j] + vocab[subword_candidate]
                elif inference_mode == "maxmin":
                    new_cost = min(costs[j], vocab[subword_candidate])
                else:
                    raise ValueError("Unkown inference mode.")
                scores.append(new_cost)
                indices.append(j)
        if not scores:
            costs[i] = -1000
            prev[i - 1] = i - 1
        else:
            if sample:
                normalizer = logsumexp(scores)
                norm_scores = [
                    math.exp(s - normalizer) for s in scores]
                idx = np.random.choice(len(scores), p=norm_scores)
            else:
                idx = max(range(len(scores)), key=lambda i: scores[i])

            costs[i] = scores[idx]
            prev[i - 1] = indices[idx]

    # Second, reconstrct the best options
    subwords = []
    idx = len(prev) - 1
    while idx >= 0:
        new_idx = prev[idx]
        #if new_idx == 0:
        #    break
        sbwrd = word[new_idx:idx + 1]
        if is_bert_wordpiece and new_idx > 0:
            sbwrd = "##" + sbwrd
        subwords.append(sbwrd)

        idx = new_idx - 1
    return list(reversed(subwords)), costs[-1]


def forward_costs(
        word: str, vocab: Dict[str, float]) -> List[str]:

    costs = [0. for _ in range(len(word) + 1)]  # for each position, this log p(prefix)
    for i in range(1, len(word) + 1):
        scores = []  # list of possible prefix scores how to get here
        for j in range(i):
            subword_candidate = word[j:i]

            if subword_candidate in vocab:
                assert vocab[subword_candidate] <= 0, "vocab should contain log probs"
                new_cost = costs[j] + vocab[subword_candidate]
                scores.append(new_cost)

        if not scores:
            #raise ValueError(f"No scores returned, no subwords of {word} in vocab - double check!")
            costs[i] = -1000
        else:
            costs[i] = logsumexp(scores)

    return costs


def backward_costs(word, vocab):
    costs = [0. for _ in range(len(word) + 1)]  # for each position, this log p(suffix)

    for begin in reversed(range(len(word))):
        scores = []

        for end in range(begin + 1, len(word) + 1):
            subword_candidate = word[begin:end]

            if subword_candidate in vocab:
                assert vocab[subword_candidate] <= 0, "vocab should contain log probs"
                new_cost = costs[end] + vocab[subword_candidate]
                scores.append(new_cost)

        if not scores:
            #raise ValueError(f"No scores returned, no subwords of {word} in vocab - double check!")
            costs[begin] = -1000
        else:
            costs[begin] = logsumexp(scores)
    return costs


def expected_counts(word, vocab):
    fw_costs = forward_costs(word, vocab)
    bw_costs = backward_costs(word, vocab)

    exp_counts = {}

    for begin in range(len(word)):

        for end in range(begin + 1, len(word) + 1):

            subword = word[begin:end]
            if subword not in vocab:
                continue

            score = fw_costs[begin] + vocab[subword] + bw_costs[end]

            if subword not in exp_counts:
                exp_counts[subword] = score
            else:
                exp_counts[subword] = logsumexp([exp_counts[subword], score])

    if not exp_counts:
        return {}

    lse = logsumexp(list(exp_counts.values()))
    return {e: val - lse for e, val in exp_counts.items()}


#print(viterbi_segment("aaabb", {"aa": -1.5, "a": -1, "b":  -6, "bb": -10}))
# (['a', 'aa', 'bb'], -12.5)

#print(viterbi_segment("aaabb", {"aa": -1.5, "a": -1, "b":  -1, "bb": -10}))
# (['a', 'aa', 'b', 'b'], -4.5)

#print(viterbi_segment("aaaab", {"aa": -1.5, "a": -1, "b":  -1, "bb": -10}))
# (['aa', 'aa', 'b'], -4.0)

#print(viterbi_segment("aaaab", {"aa": -1.5, "a": -1, "b":  -1, "bb": -10}, sample=True))
