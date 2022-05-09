# coding=utf-8
import dataclasses
import random
from typing import Any

from torch.utils.data import Dataset

from highball.config import DatasetConfig
from highball.vocab import SimpleVocabulary


@dataclasses.dataclass
class ReverseToyDatasetConfig(DatasetConfig):
    vocab: SimpleVocabulary
    data_cnt: int
    max_seq_len: int

    def instantiate(self) -> Dataset:
        return ReverseToyDataset(self.vocab, self.data_cnt, self.max_seq_len)


class ReverseToyDataset(Dataset):

    def __init__(
            self,
            vocab: SimpleVocabulary,
            data_cnt: int,
            max_seq_len: int,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        # self.vocab = SimpleVocabulary(
        #     [chr(e) for e in range(ord('a'), ord('z'))] +
        #     [chr(e) for e in range(ord('A'), ord('Z'))] +
        #     [chr(e) for e in range(ord('0'), ord('9'))]
        # )
        self.vocab = vocab
        first_idx = 4
        last_idx = len(self.vocab) - 1

        eos_token_id = self.vocab.idx(self.vocab.DEFAULT_EOS_TOKEN)

        self.data = []
        for i in range(data_cnt):
            seq_len = random.randint(1, self.max_seq_len - 1)
            seq = [random.randint(first_idx, last_idx) for _ in range(seq_len)]
            item = {
                'src_seq': seq,
                'tgt_seq': [e for e in reversed(seq)]
            }
            item['src_seq'].append(eos_token_id)
            item['tgt_seq'].append(eos_token_id)

            self.data.append(item)

    def __getitem__(self, index: Any):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def make_default_vocab() -> SimpleVocabulary:
        return SimpleVocabulary(
            [chr(e) for e in range(ord('a'), ord('z'))] +
            [chr(e) for e in range(ord('A'), ord('Z'))] +
            [chr(e) for e in range(ord('0'), ord('9'))]
        )
