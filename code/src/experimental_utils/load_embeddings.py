import torch
import torch.nn as nn
from torchtext.vocab import Vocab

from core.models import CBOW

def load_joint_embeddings(
    vocab: Vocab, embedding_dim: int, device: str
) -> nn.Embedding:
    """Load joint embeddings from a pretrained cBOW model."""

    cbow = CBOW(vocab_size=len(vocab), embedding_dim=embedding_dim)
    cbow.load_state_dict(
        torch.load(
            f"../../model_checkpoint/cbow_joint_{embedding_dim}.pt", map_location=device
        )
    )
    for param in cbow.parameters():
        param.requires_grad = False
    embeddings = cbow.embeddings
    return embeddings


def load_separate_embeddings(
    vocabs: list[Vocab], embedding_dim: int, modalities: list, device: str
) -> nn.ModuleList:
    """Load separate embeddings for each modaltiy from pretrained cBOW models."""
    embeddings = nn.ModuleList()
    for j, vocab in enumerate(vocabs):
        temp_cbow = CBOW(vocab_size=len(vocab), embedding_dim=embedding_dim)
        temp_cbow.load_state_dict(
            torch.load(
                f"../../model_checkpoint/cbow_{modalities[j]}_{embedding_dim}.pt",
                # f"../../model_checkpoint/cbow_{modalities[j]}.pt",
                map_location=device,
            )
        )
        for param in temp_cbow.parameters():
            param.requires_grad = False

        embeddings.append(temp_cbow.embeddings)
    return embeddings