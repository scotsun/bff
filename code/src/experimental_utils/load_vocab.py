from tqdm import tqdm
import pandas as pd
from torchtext.vocab import Vocab
from torchtext.transforms import VocabTransform

from core.data_utils import Corpus, build_vocab

def load_joint_vocab(test_cohort) -> tuple[Vocab, VocabTransform]:
    """Load the joint vocab for all medical records and exclude the test cohort."""
    fpaths = []
    for i in range(2015, 2023):
        fpaths += [
            f"../../data/pretrain/processed/{i}/birth_newborn.csv",
            f"../../data/pretrain/processed/{i}/birth_mom.csv",
            f"../../data/pretrain/processed/{i}/developmental.csv",
            f"../../data/pretrain/processed/{i}/prenatal.csv",
        ]
    corpus_list = []
    for f in tqdm(fpaths, unit="file"):
        temp_df = pd.read_csv(
            f, dtype={"PATID": str, "ENCID": str, "NEWBORN_PATID": str}
        )
        temp_df = temp_df.loc[~temp_df["PATID"].isin(test_cohort["PATID"])]
        temp_df["EVENT"] = temp_df["EVENT"].astype("category")
        corpus = Corpus(temp_df.groupby("PATID")["EVENT"].apply(list).to_list())
        corpus_list.append(corpus)

    vocab = build_vocab(corpus_list=corpus_list, min_freq=5)
    print("vocab size:", len(vocab))
    transform = VocabTransform(vocab)
    return vocab, transform


def load_separate_vocab(
    test_cohort, modalities: list
) -> tuple[list[Vocab], list[VocabTransform]]:
    """Load separate vocabs for each modality in a list."""
    vocabs = []
    for m in modalities:
        # for each modality build a vocab throughout the time interval
        fpaths = []
        for i in range(2015, 2023):
            fpaths.append(f"../../data/pretrain/processed/{i}/{m}.csv")
        corpus_list = []
        for f in tqdm(fpaths, unit="file"):
            temp_df = pd.read_csv(
                f, dtype={"PATID": str, "ENCID": str, "NEWBORN_PATID": str}
            )
            temp_df = temp_df.loc[~temp_df["PATID"].isin(test_cohort["PATID"])]
            temp_df["EVENT"] = temp_df["EVENT"].astype("category")
            corpus = Corpus(temp_df.groupby("PATID")["EVENT"].apply(list).to_list())
            corpus_list.append(corpus)
        vocab = build_vocab(corpus_list=corpus_list, min_freq=5)
        print(m, len(vocab))
        vocabs.append(vocab)
    transform = [VocabTransform(vocab) for vocab in vocabs]
    return vocabs, transform