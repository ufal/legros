
from typing import List, Dict


def viterbi_segment(
        word: str, vocab: Dict[str, float], sample: bool = False) -> List[str]:

    costs = [0. for _ in range(len(word) + 1)]
    prev = [0 for _ in word]

    # First, dynamic programming
    for i in range(1, len(word) + 1):
        scores = []
        indices = []
        for j in range(i):
            subword_candidate = word[j:i]
            if subword_candidate in vocab:
                new_cost = costs[j] + vocab[subword_candidate]
                scores.append(new_cost)
                indices.append(j)
        if not scores:
            costs[i] = -1000
            prev[i - 1] = i - 1
        else:
            if sample:
                pass
                # norm_scores =[
                #     exp(s - LogExpFunctions.logsumexp(scores)) for s in scores]
                # threshold = rand()
                # idx = 0
                # cum_prob = 0.0
                # for i, p in enumerate(norm_scores):
                #     idx = i
                #     cum_prob += p
                #     if p > threshold:
                #         break
            else:
                idx = max(range(len(scores)), key=lambda i: scores[i])

            costs[i] = scores[idx]
            prev[i - 1] = indices[idx]
    #import ipdb; ipdb.set_trace()

    # Second, reconstrct the best options
    subwords = []
    idx = len(prev) - 1
    while idx >= 0:
        new_idx = prev[idx]
        #if new_idx == 0:
        #    break
        subwords.append(word[new_idx:idx + 1])
        idx = new_idx - 1
    return list(reversed(subwords)), costs[-1]


#print(viterbi_segment("aaabb", {"aa": -1.5, "a": -1, "b":  -6, "bb": -10}))
# (['a', 'aa', 'bb'], -12.5)

#print(viterbi_segment("aaabb", {"aa": -1.5, "a": -1, "b":  -1, "bb": -10}))
# (['a', 'aa', 'b', 'b'], -4.5)

#print(viterbi_segment("aaaab", {"aa": -1.5, "a": -1, "b":  -1, "bb": -10}))
# (['aa', 'aa', 'b'], -4.0)

