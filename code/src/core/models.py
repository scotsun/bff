"""Word embedding models."""

import math
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


class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        positional_embedding = self.pe[position]
        return positional_embedding


class BERT(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, nhead, hidden_size, n_layer, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoder(embed_size, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.token_pred_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens, positions, pad_masks):
        token_embedded = self.token_embedding(tokens)
        position_embedded = self.pos_encoder(positions)
        embedded_sources = token_embedded + position_embedded

        embedded_sources = self.transformer_encoder(embedded_sources, src_key_padding_mask=pad_masks)
        token_predictions = self.token_pred_head(embedded_sources)
        return token_predictions
