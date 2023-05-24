#!/usr/bin/env python3

from typing import List
import argparse
from collections import defaultdict
import sys


def pretokenize(text: str) -> List[str]:
    text = "▁" + text.replace(" ", "▁")
    tokens = []
    token_start = 0
    for i, char in enumerate(text):
        if i == 0:
            continue

        if text[i - 1] != '▁' and (
                not char.isalnum() or not text[i - 1].isalnum()):
            tokens.append(text[token_start:i])
            token_start = i
    tokens.append(text[token_start:])
    return tokens
