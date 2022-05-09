
class Vocab:
    def __init__(self, wordlist):
        self.wordlist = wordlist

        if not "###" in wordlist:
            wordlist.insert(0, "###")

        self.max_subword_length = max(len(v) for v in wordlist)
        self.size = len(wordlist)

        self.unique_tokens = set(wordlist)

        self.word2idx = {w: i for i, w in enumerate(wordlist)}

    def __contains__(self, item):
        return item in self.unique_tokens
