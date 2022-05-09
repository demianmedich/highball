# coding=utf-8
from typing import Iterable, Optional, List


def load_vocabulary(
        vocab_path: str,
        special_tokens: Optional[List[str]] = None,
) -> "SimpleVocabulary":
    with open(vocab_path, encoding='utf-8', mode='rt') as f:
        lines = f.readlines()

    tokens = [line.rstrip().split()[0] for line in lines]
    return SimpleVocabulary(tokens, special_tokens)


class SimpleVocabulary:
    """Dictionary class for token and index pairs"""

    DEFAULT_PAD_TOKEN = '[PAD]'
    DEFAULT_BOS_TOKEN = '[BOS]'
    DEFAULT_EOS_TOKEN = '[EOS]'
    DEFAULT_UNK_TOKEN = '[UNK]'

    def __init__(
            self,
            tokens: List[str],
            special_tokens: Optional[List[str]] = None,
            use_special_tokens: bool = True,
    ) -> None:
        """Dictionary class constructor

        Args:
            tokens: Iterable class instance which has tokens
            special_tokens: (Opt) Iterable class instance which has special tokens
                default tokens are ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
            use_special_tokens: If false and special_tokens is also None, special tokens will not be used.
        """

        if special_tokens is not None and len(special_tokens) > 0:
            use_special_tokens = True

        if special_tokens is None and use_special_tokens:
            special_tokens = [
                self.DEFAULT_PAD_TOKEN,
                self.DEFAULT_BOS_TOKEN,
                self.DEFAULT_EOS_TOKEN,
                self.DEFAULT_UNK_TOKEN
            ]

        self.str2idx_dict = {token: i for i, token in enumerate(special_tokens + tokens)}
        self.idx2str_dict = {v: k for k, v in self.str2idx_dict.items()}

    def token_to_idx(self, seq: Iterable[str]) -> List[int]:
        return [self.str2idx_dict[token] for token in seq]

    def idx_to_token(self, indices: Iterable[int]) -> List[str]:
        return [self.idx2str_dict[idx] for idx in indices]

    def token(self, idx: int):
        return self.idx2str_dict[idx]

    def idx(self, token: str):
        return self.str2idx_dict[token]

    def __len__(self):
        return len(self.str2idx_dict)
