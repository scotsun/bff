"""Data utilility functions."""

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.transforms import VocabTransform
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Generator

from core.utils.tokenizer_utils import (
    PAD_TOKEN,
    UNK_TOKEN,
    CBOW_SPECIAL_TOKENS,
)


class Corpus(Dataset):
    def __init__(self, data, max_length: int | None = None) -> None:
        self.data = data
        self.max_length = max_length

    def __getitem__(self, index) -> list:
        if self.max_length is not None:
            return self.data[index][: self.max_length]
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GloVeDataset(Dataset):
    def __init__(self, cooccurrance: defaultdict) -> None:
        super().__init__()
        self.cooccurrance: list = list(cooccurrance.items())

    def __getitem__(self, index) -> tuple:
        return self.cooccurrance[index]

    def __len__(self) -> int:
        return len(self.cooccurrance)


def tensor_pad(tensor: torch.Tensor, max_len: int, padding_value: int):
    padded_tensor = pad_sequence(tensor, batch_first=True, padding_value=padding_value)
    if padded_tensor.shape[1] < max_len:
        extra_padding = torch.zeros(
            padded_tensor.shape[0], max_len - padded_tensor.shape[1]
        )
        padded_tensor = torch.cat([padded_tensor, extra_padding], dim=1)
    if padded_tensor.shape[1] == 0:
        padded_tensor = torch.zeros(padded_tensor.shape[0], max_len)
    return padded_tensor[:, :max_len]


class MultiModalDataGenerator:
    """
    Helper class that generates data beforehand so that the __getitem__ is fast in MultiMataDataset.
    """

    def __init__(
        self,
        modalities: list[pd.DataFrame],
        patid_list: np.ndarray,
        outcomes: list[np.ndarray],
        transform: VocabTransform | list[VocabTransform],
        max_length: int,
    ) -> None:
        self.modalities = modalities
        self.patid_list = patid_list
        self.outcomes = outcomes
        self.transform = transform
        self.max_length = max_length

    def __getitem__(self, index) -> tuple:
        patid = self.patid_list[index]
        tokens = []  # list of tokens from different modalities
        for j, modality in enumerate(self.modalities):
            try:  # `r` stands for record
                r = modality.get_group(patid)["EVENT"].to_list()[: self.max_length]
                if isinstance(self.transform, VocabTransform):
                    r = self.transform(r)
                elif isinstance(self.transform, list):
                    r = self.transform[j](r)
                else:
                    raise ValueError("incorrect `transform` value.")
            except KeyError:
                r = []
            tokens.append(r)
        masks = [1 if len(r) > 0 else 0 for r in tokens]
        y = [outcome[index] for outcome in self.outcomes]
        # break the list of tokens
        return *tokens, masks, *y

    def __len__(self) -> int:
        return len(self.patid_list)


class MultiModalDataset(Dataset):
    """
    Multi-modality dataset
    """

    def __init__(
        self,
        multimodal_generator: MultiModalDataGenerator,
        complete_case: bool,
        n_outcomes: int = 1,
    ) -> None:
        super().__init__()
        self.multimodal_generator = multimodal_generator
        self.data = []
        for i in tqdm(range(len(self.multimodal_generator))):
            elem = self.multimodal_generator[i]
            if complete_case:
                if all(
                    [b == 1 for b in elem[-(n_outcomes + 1)]]
                ):  # check complete case
                    self.data.append(elem)
                else:
                    continue
            else:
                if all(
                    [b == 0 for b in elem[-(n_outcomes + 1)]]
                ):  # ignore if missing all
                    continue
                else:
                    self.data.append(elem)

    def __getitem__(self, index) -> tuple:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class FilteredMultiModalDataset(Dataset):
    def __init__(self, dataset: MultiModalDataset, mask_pattern: list) -> None:
        super().__init__()
        self.filtered_data = list(
            filter(lambda elem: elem[-3] == mask_pattern, tqdm(dataset))
        )

    def __getitem__(self, index) -> tuple:
        return self.filtered_data[index]

    def __len__(self) -> int:
        return len(self.filtered_data)


class MultiModalCollate:
    """
    Return batch in a dict data struct
    {
        padded_tokens: list[tensor],
        mask: tensor[bool],
        event: tensor[bool],
        [Optional] time: tensor[float],
    }
    """

    def __init__(self, n_modality, padding_token=0, max_len=512, survival=True):
        self.n_modality = n_modality
        self.padding_token = padding_token
        self.max_len = max_len
        self.survival = survival

    def __call__(self, batch) -> dict:
        zipped_list = list(zip(*batch))
        padded_tokens = []
        for i in range(self.n_modality):
            token = zipped_list[i]
            padded_token = pad_sequence(
                [torch.tensor(r) for r in token],
                batch_first=True,
                padding_value=self.padding_token,
            )
            if (
                padded_token.shape[1] < self.max_len
            ):  # in case of requiring extra padding
                extra_padding = torch.zeros(
                    padded_token.size(0), self.max_len - padded_token.size(1)
                )
                padded_token = torch.cat([padded_token, extra_padding], dim=1)
            if padded_token.size(1) == 0:
                # in case that the entire batch is missing a modality
                padded_token = torch.zeros(padded_token.size(0), self.max_len)
            padded_tokens.append(padded_token)
        mask = torch.tensor(zipped_list[self.n_modality], dtype=torch.float)
        event = torch.tensor(zipped_list[self.n_modality + 1], dtype=torch.long)

        batch_result = {
            "inputs": torch.stack(padded_tokens, dim=1).long(),
            "mask": mask,
            "event": event,
        }
        if self.survival:
            time = torch.tensor(zipped_list[self.n_modality + 2], dtype=torch.float)
            batch_result["time"] = time

        return batch_result


def build_corpus(fpath: str):
    """Build Corpus object given a file path to a csv file."""
    df = pd.read_csv(fpath, encoding="latin1", dtype={"PATID": str})
    df["EVENT"] = df["EVENT"].astype("category")
    corpus = Corpus(df.groupby("PATID")["EVENT"].apply(list).to_list())
    return corpus


def build_corpus_from_df(df: pd.DataFrame, max_length: int):
    """Build Corpus object given a dataframe."""
    df["EVENT"] = df["EVENT"].astype("category")
    corpus = Corpus(df.groupby("PATID")["EVENT"].apply(list).to_list(), max_length)
    return corpus


def build_vocab(corpus_list: list[Corpus], min_freq: int, mode: str = "cbow"):
    """Build vocabulary based on a list of Corpus."""
    match mode:
        case "cbow":
            special_tokens = CBOW_SPECIAL_TOKENS
        case _:
            ValueError("only support cbow for now")
    total_corpus = ConcatDataset(corpus_list)
    vocab = build_vocab_from_iterator(
        total_corpus,
        min_freq=min_freq,
        specials=list(special_tokens.keys()),
        special_first=True,
    )  # this assign {<PAD>: 0, <UNK>: 1}
    vocab.set_default_index(vocab[UNK_TOKEN])  # unseen token will be signed to <UNK>
    return vocab


def _window(token: list[int], start_idx: int, end_idx: int) -> list[int]:
    # inclusive interval
    last_index = len(token)
    window = token[max(0, start_idx) : min(last_index, end_idx + 1)]
    return window


def context_windows(
    token: list[str],
    left_size: int,
    right_size: int,
    vocab: Vocab,
    padding: bool = False,
) -> Generator[tuple[list[int], int, list[int]], None, None]:
    """Create context windows & convert the tokens to indices."""
    pad_id = vocab.lookup_indices([PAD_TOKEN])[0]
    transform = torchtext.transforms.VocabTransform(vocab)

    for i, target in enumerate(token):
        start_idx = i - left_size
        end_idx = i + right_size
        left_context = transform(_window(token, start_idx, i - 1))
        right_context = transform(_window(token, i + 1, end_idx))
        target = transform([target])[0]
        if padding:
            _left_pad = left_size - len(left_context)
            left_context = [pad_id] * _left_pad + left_context

            _right_pad = right_size - len(right_context)
            right_context = right_context + [pad_id] * _right_pad
        yield left_context, target, right_context


# modality list [baby_birth, baby_dev, mom_birth, mom_prenatal]
MODALITY_DATA_SELECT: dict = {
    "final_check": lambda m: torch.ones(m.size(0)).bool(),
    "mid_check": lambda m: (m[:, [0, 2, 3]] == 1).any(dim=1),
    "first_check": lambda m: m[:, -1] == 1,
}

SUPPRESS_MODALITY: dict = {
    "final_check": lambda m: m,
    "mid_check": lambda m: m.index_fill_(1, torch.tensor(1).to(m.device), 0),
    "first_check": lambda m: m.index_fill_(1, torch.tensor([0, 1, 2]).to(m.device), 0),
}
