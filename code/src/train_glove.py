import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from core.data_utils import GloVeDataset, build_vocab, build_corpus
from core.models import GloVe, compute_co_occur_matrix

EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    corpus1 = build_corpus("../../data/processed/birth_newborn.csv")
    corpus2 = build_corpus("../../data/processed/birth_mom.csv")
    vocab = build_vocab(corpus_list=[corpus1, corpus2])

    corpus = ConcatDataset([corpus1, corpus2])
    glove_dataset = GloVeDataset(
        cooccurrance=compute_co_occur_matrix(corpus, 6, 6, vocab)
    )
    dataloader = DataLoader(glove_dataset, batch_size=128, shuffle=True)

    model = GloVe(embed_size=256, vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(EPOCHS):
        total_loss = 0
        total_batch = 0
        with tqdm(dataloader, unit="batch") as bar:
            bar.set_description(f"epoch {epoch}")
            for (focal, context), count in bar:
                focal, context, count = (
                    focal.to(DEVICE),
                    context.to(DEVICE),
                    count.to(DEVICE),
                )
                count = count.float()
                optimizer.zero_grad()
                loss = model(focal, context, count).mean()
                loss.backward()
                optimizer.step()

                # update running stat
                total_loss += loss.item()
                total_batch += 1
                bar.set_postfix(loss=total_loss / total_batch)

    torch.save(model.state_dict(), "../../model_checkpoint/glove_example.pt")


if __name__ == "__main__":
    main()
