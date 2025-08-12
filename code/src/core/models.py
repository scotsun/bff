"""Word embedding models."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from tqdm import tqdm
from collections import defaultdict

from core.data_utils import context_windows


class CBOW(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_maxnorm: float = 1.0,
    ) -> None:
        """
        Continuous bag of word.
        """
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
            max_norm=embedding_maxnorm,
        )
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        c_embedding = self.embeddings(x)
        h = torch.mean(c_embedding, dim=1)
        out = self.linear(h)
        return out


class GloVe(nn.Module):
    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        min_occurance: int = 1,
        x_max: int = 100,
        alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.x_max = x_max
        self.alpha = alpha
        self.min_occurrance = min_occurance
        self.vocab_size = vocab_size

        self.focal_embedding = nn.Embedding(vocab_size, embed_size)
        self.context_embedding = nn.Embedding(vocab_size, embed_size)
        self.focal_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        for param in self.parameters():
            nn.init.xavier_normal_(param)

    def forward(self, focal_input, context_input, cooccurance_count):
        """Forward pass to calculate loss"""
        # get embedding
        focal_embed = self.focal_embedding(focal_input)
        context_embed = self.context_embedding(context_input)
        focal_bias = self.focal_biases(focal_input)
        context_bias = self.context_biases(context_input)
        # compute loss
        weight_factor = torch.pow(cooccurance_count / self.x_max, self.alpha)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_coocurrances = torch.log(cooccurance_count)

        loss = (
            weight_factor
            * (embedding_products + focal_bias + context_bias - log_coocurrances) ** 2
        )
        return loss

    @property
    def embeddings(self):
        return (
            self.focal_embedding.weight.detach()
            + self.context_embedding.weight.detach()
        )


def compute_co_occur_matrix(
    corpus: Dataset, left_size: int, right_size: int, vocab: Vocab
):
    """
    Compute context-window-based co-occurance matrix of corpus.
    """
    # token_counter = Counter()
    co_occur_counts = defaultdict(float)
    for record in tqdm(corpus, desc="compute co-occurance matrix"):
        # token_counter.update(record)
        for left_context, target_token, right_context in context_windows(
            record, left_size, right_size, vocab
        ):
            for i, context_token in enumerate(left_context[::-1]):
                # i is the positional distance between target_token and context_token
                co_occur_counts[(target_token, context_token)] += 1 / (i + 1)
            for i, context_token in enumerate(right_context):
                co_occur_counts[(target_token, context_token)] += 1 / (i + 1)
    return co_occur_counts
