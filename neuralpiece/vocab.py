
class Vocab:
    def __init__(self, wordlist):
        self.wordlist = wordlist

        self.max_subword_length = max(len(v) for v in wordlist)
        self.size = len(wordlist)

        self.unique_tokens = set(wordlist)

    def __contains__(self, item):
        return item in self.unique_tokens
