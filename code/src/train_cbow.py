"""Train scratch file will be replaced by a pytorch_lightly or trainer object."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import pandas as pd
import argparse

from core.data_utils import Corpus, build_vocab, context_windows
from core.models import CBOW


def main():
    parser = argparse.ArgumentParser(description="Train file for Word2Vec.")
    parser.add_argument(
        "--embeddingDim",
        type=int,
        default=256,
        help="int: embedding size; default 256",
    )
    parser.add_argument("--epoch", type=int, default=1, help="int: number of epochs")
    parser.add_argument(
        "--separateEmbedding", action="store_true", help="flag indicator"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="joint",
        help="str: modality name {joint, birth_newborn, birth_mom, developmental, prenatal}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="str: indicating device name",
    )
    args = parser.parse_args()

    EMBEDDING_DIM = args.embeddingDim
    EPOCHS = args.epoch
    MODALITY = args.modality
    DEVICE = args.device
    print(DEVICE)
    if args.separateEmbedding and MODALITY == "joint":
        raise ValueError("Contradiction in training args.")

    test_cohort = pd.read_csv("../../data/outcome/test_asd.csv", dtype={"PATID": str})[
        "PATID"
    ].unique()

    fpaths = []
    for i in range(2015, 2023):
        if args.separateEmbedding:
            fpaths += [f"../../data/pretrain/processed/{i}/{MODALITY}.csv"]
        else:
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
        temp_df = temp_df.loc[~temp_df["PATID"].isin(test_cohort)]
        temp_df["EVENT"] = temp_df["EVENT"].astype("category")
        corpus = Corpus(temp_df.groupby("PATID")["EVENT"].apply(list).to_list())
        corpus_list.append(corpus)

    vocab = build_vocab(corpus_list=corpus_list, min_freq=5)
    print("vocab size:", len(vocab))
    
    corpus = ConcatDataset(corpus_list)

    dataset = []
    for record in tqdm(corpus, desc="build context window"):
        for lc, t, rc in context_windows(record, 6, 6, vocab, padding=True):
            dataset.append((t, torch.tensor(lc + rc, dtype=torch.long)))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = CBOW(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_acc, total_loss, n_batch = 0.0, 0.0, 0.0
        with tqdm(dataloader, unit="batch") as bar:
            bar.set_description(f"epoch {epoch}")
            for target, context in bar:
                target, context = target.to(DEVICE), context.to(DEVICE)
                optimizer.zero_grad()
                logits = model(context)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                # update running stat
                _acc = (logits.argmax(dim=1) == target).float().mean().item()
                _loss = loss.item()

                total_acc += _acc
                total_loss += _loss
                n_batch += 1

                bar.set_postfix(loss=total_loss / n_batch, cbow_acc=total_acc / n_batch)

    torch.save(
        model.state_dict(), f"../../model_checkpoint/cbow_{MODALITY}_{EMBEDDING_DIM}.pt"
    )


if __name__ == "__main__":
    main()
