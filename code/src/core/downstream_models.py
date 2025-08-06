import torch
import torch.nn as nn
from torch.nn import ModuleList
from core.losses import pairwise_cosine, snn_loss


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.num_layers = 2
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers=2, nonlinearity="relu", batch_first=True
        )

    def forward(self, x, mask):
        # x: [B, L, E]; mask: [B, L]
        batch_size, seq_len = mask.shape

        h = torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size).to(x.device)
        outs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, E]
            mask_t = mask[:, t].unsqueeze(-1)  # [B, 1]
            x_t = x_t * mask_t
            out, h = self.rnn(x_t.unsqueeze(1), h)
            outs.append(out.squeeze(1))

        return torch.stack(outs, dim=1), h


class ForecastingAE(nn.Module):
    def __init__(
        self,
        embedding_module,
        embed_size,
        enc_size: int = 128,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding_module = embedding_module
        self.enc_size = enc_size
        if freeze_embeddings:
            for param in embedding_module.parameters():
                param.requires_grad = False
        self.enc = RNNEncoder(embed_size, enc_size)
        self.dec = nn.Sequential(
            nn.Linear(enc_size, 128), nn.ReLU(), nn.Linear(128, embed_size)
        )

    def masked_avg_pool(self, token, embedding, pad_token: int = 0):
        """
        Avg pool and remove padding token's influence.

        token : (B, n_modality, L) or (B, L)
        embedding: (B, n_modality, L, D) or (B, L, D)
        """
        nonpad_mask = (token != pad_token).float()
        # zero out pad_token's representation
        embedding = embedding * nonpad_mask.unsqueeze(-1)
        pooled = embedding.sum(dim=-2) / nonpad_mask.sum(dim=-1, keepdim=True).clamp(
            min=1
        )
        # edge case for pooled: zeros for a missing modality
        return pooled

    def encode(self, x, masks):
        _, h = self.enc(x, masks)
        return h[-1]

    def forward(self, inputs, masks):
        if isinstance(self.embedding_module, ModuleList):  # seperate embedding layers
            raise ValueError("should be joint")
        else:  # joint embedding layer
            embeddings = self.embedding_module(inputs)
        # ordering modality by time
        x_past = self.masked_avg_pool(inputs[:, [3, 2, 0]], embeddings[:, [3, 2, 0]])
        masks_past = masks[:, [3, 2, 0]]
        # print(masks_past)
        # raise ValueError
        x_future = self.masked_avg_pool(inputs[:, 1], embeddings[:, 1])
        last_h = self.encode(x_past, masks_past)
        x_future_hat = self.dec(last_h)  # use last hidden state as the bottleneck
        return x_future_hat, x_future, last_h


class SimpleTimeNN(nn.Module):
    def __init__(self, enc_size: int, n_bins: int):
        super().__init__()

        self.clf = nn.Sequential(
            nn.Linear(enc_size, enc_size),
            nn.ReLU(),
            nn.Linear(enc_size, n_bins + 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, h):
        outputs = self.clf(h).squeeze()
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
        )
        self.z_content = nn.Linear(512, int(output_size / 2), bias=False)
        self.z_style = nn.Linear(512, int(output_size / 2), bias=False)

    def forward(self, x):
        h = self.mlp(x)
        return self.z_content(h), self.z_style(h)


class MaskedAvg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, proj_list, masks):
        proj_hs = torch.stack(proj_list, dim=1)  # -> (B, m, D)
        w = masks.unsqueeze(-1).expand(-1, -1, proj_hs.shape[-1]) / masks.sum(
            dim=1, keepdim=True
        ).unsqueeze(-1)
        masks = masks.view(-1, proj_hs.shape[1], 1).repeat(1, 1, proj_hs.shape[-1])
        mixed = (proj_hs * masks).sum(dim=1) / masks.sum(dim=1)
        return mixed, w


class SelfAttnMixer(nn.Module):
    def __init__(self, clf_hidden_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.Wq = nn.Linear(clf_hidden_size, d_k, bias=False)
        self.Wk = nn.Linear(clf_hidden_size, d_v, bias=False)
        self.Wv = nn.Linear(clf_hidden_size, d_v, bias=False)

    def forward(self, proj_list, masks):
        proj_hs = torch.stack(proj_list, dim=1)  # -> (B, m, D)
        Q, K, V = self.Wq(proj_hs), self.Wk(proj_hs), self.Wv(proj_hs)
        attn_score = torch.bmm(Q, K.transpose(1, 2)) / self.d_k
        attn_score = attn_score.masked_fill(masks.unsqueeze(1) == 0, -999)
        attn_w = attn_score.softmax(dim=2)  # -> (B, m, m)

        mixed = torch.bmm(attn_w, V).mean(dim=1)
        return mixed, attn_w


class SelfGatingMixer(nn.Module):
    def __init__(self, clf_hidden_size, r=1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(clf_hidden_size, int(clf_hidden_size / r)),
            nn.ReLU(),
            nn.Linear(int(clf_hidden_size / r), clf_hidden_size),
        )

    def forward(self, proj_list, masks):
        proj_hs = torch.stack(proj_list, dim=1)  # -> (B, m, D)
        gate_logits = self.gate(torch.stack(proj_list, dim=1))

        masks = masks.view(-1, proj_hs.shape[1], 1).repeat(1, 1, proj_hs.shape[-1])
        # rule out missing modality in mixing process
        gate_logits = gate_logits * masks + (-999) * (1 - masks)
        softmax_w = gate_logits.softmax(dim=1)
        mixed = (proj_hs * softmax_w).sum(dim=1)

        return mixed, softmax_w


class BackboneModel(nn.Module):
    def __init__(
        self,
        embedding_module: nn.Embedding | ModuleList,
        n_modality: int,
        embed_size: int,
        zp_only: bool,
        mixing_module: str = "softmax-gating",
        enc_size: int = 128,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_module = embedding_module

        self.n_modality = n_modality
        self.enc_size = enc_size
        # freeze embedding
        if freeze_embeddings:
            for param in embedding_module.parameters():
                param.requires_grad = False
        # nonlinear projection heads
        self.encoders = nn.ModuleList(
            [Encoder(embed_size, enc_size) for _ in range(n_modality)]
        )
        # downstream clf
        self.zp_only = zp_only
        clf_hidden_size = int(enc_size / 2) if zp_only else enc_size

        match mixing_module:
            case "softmax-gating":
                self.mixing_module = SelfGatingMixer(clf_hidden_size)
            case "self-attention":
                self.mixing_module = SelfAttnMixer(
                    clf_hidden_size, clf_hidden_size, clf_hidden_size
                )
            case "masked-avg":
                self.mixing_module = MaskedAvg()
            case _:
                raise ValueError("undefined fusion technique")
        self.clf = nn.Sequential(
            nn.Linear(clf_hidden_size, clf_hidden_size),
            nn.ReLU(),
            nn.Linear(clf_hidden_size, 1),
            nn.Sigmoid(),
        )

    def masked_avg_pool(self, token, embedding, pad_token: int = 0):
        """
        Avg pool and remove padding token's influence.

        token : (B, n_modality, L) or (B, L)
        embedding: (B, n_modality, L, D) or (B, L, D)
        """
        nonpad_mask = (token != pad_token).float()
        # zero out pad_token's representation
        embedding = embedding * nonpad_mask.unsqueeze(-1)
        pooled = embedding.sum(dim=-2) / nonpad_mask.sum(dim=-1, keepdim=True).clamp(
            min=1
        )
        # edge case for pooled: zeros for a missing modality
        return pooled

    def forward(self, inputs, masks):
        if isinstance(self.embedding_module, ModuleList):  # seperate embedding layers
            embedding_list = [
                self.embedding_module[j](inputs[:, j]) for j in range(self.n_modality)
            ]
        else:  # joint embedding layer
            _embeddings = self.embedding_module(inputs)
            embedding_list = [_embeddings[:, j] for j in range(self.n_modality)]
        proj_c_list = [
            self.encoders[j](self.masked_avg_pool(inputs[:, j], e))[0]
            for j, e in enumerate(embedding_list)
        ]
        proj_t_list = [
            self.encoders[j](self.masked_avg_pool(inputs[:, j], e))[1]
            for j, e in enumerate(embedding_list)
        ]
        if self.zp_only:
            mixed, w = self.mixing_module(proj_c_list, masks)
            outputs = self.clf(mixed)
        else:
            proj_list = [
                torch.cat(
                    [
                        # c:
                        proj_c_list[j],
                        # t:
                        proj_t_list[j],
                    ],
                    dim=-1,
                )
                for j in range(self.n_modality)
            ]  # list of (B, D)
            mixed, w = self.mixing_module(proj_list, masks)
            outputs = self.clf(mixed)
        return outputs, proj_c_list, proj_t_list, w


class DiscreteTimeNN(BackboneModel):
    def __init__(
        self,
        n_bins: int,
        embedding_module: nn.Embedding | ModuleList,
        n_modality: int,
        embed_size: int,
        zp_only: bool,
        mixing_module: str = "softmax-gating",
        enc_size: int = 128,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__(
            embedding_module,
            n_modality,
            embed_size,
            zp_only,
            mixing_module,
            enc_size,
            freeze_embeddings,
        )
        clf_hidden_size = int(enc_size / 2) if zp_only else enc_size

        self.clf = nn.Sequential(
            nn.Linear(clf_hidden_size, clf_hidden_size),
            nn.ReLU(),
            nn.Linear(clf_hidden_size, n_bins + 1),
            nn.Softmax(dim=-1),
        )


class MultiModalSNN(nn.Module):
    def __init__(self, device: str = "cpu", tau_init: float = 0.2) -> None:
        super().__init__()
        self.log_tau_instance = nn.Parameter(torch.tensor(tau_init).log())
        self.log_tau_modality = nn.Parameter(torch.tensor(tau_init).log())
        self.device = device

    def forward(
        self,
        z_cs: list[torch.Tensor],
        z_ts: list[torch.Tensor],
        modality_mask: torch.Tensor,
    ):
        batch_size, n_modality = modality_mask.shape
        m = modality_mask.T.reshape(-1)  # order by modality
        z_c = torch.cat(
            z_cs, dim=0
        )  # cat s.t. order by modality ie. [z1,...,z1,z2,...,z2]
        z_t = torch.cat(z_ts, dim=0)
        zc_selected, zt_selected = z_c[m.bool()], z_t[m.bool()]
        # instance level contrast:
        # pair mat
        _batch_id = torch.arange(0, batch_size).to(self.device)
        _batch_id = _batch_id.repeat(n_modality)[m.bool()]
        pos_target = (_batch_id[None, :] == _batch_id[:, None]).float()
        # sim mat
        sim = pairwise_cosine(zc_selected)
        loss_instance = snn_loss(sim, pos_target, torch.exp(self.log_tau_instance))

        # modality level contrast:
        # pair mat
        _modality_id = torch.arange(0, n_modality).to(self.device)
        _modality_id = _modality_id.repeat_interleave(batch_size)[m.bool()]
        pos_target = (_modality_id[None, :] == _modality_id[:, None]).float()
        # sim mat
        sim = pairwise_cosine(zt_selected)
        loss_modality = snn_loss(sim, pos_target, torch.exp(self.log_tau_modality))

        return (
            loss_instance[loss_instance.isfinite()].mean(),
            loss_modality[loss_modality.isfinite()].mean(),
        )


class DiscreteFailureTimeNLL(nn.Module):
    def __init__(self, bin_boundaries, tolerance=1e-8, device="cpu"):
        super().__init__()
        self.bin_starts = torch.tensor(bin_boundaries[:-1]).to(device)
        self.bin_ends = torch.tensor(bin_boundaries[1:]).to(device)
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.tolerance = tolerance

    def _discretize_times(self, times):
        return (times[:, None] > self.bin_starts[None, :]) & (
            times[:, None] <= self.bin_ends[None, :]
        )

    def _get_proportion_of_bins_completed(self, times):
        return torch.maximum(
            torch.minimum(
                (times[:, None] - self.bin_starts[None, :]) / self.bin_lengths[None, :],
                torch.tensor(1),
            ),
            torch.tensor(0),
        )

    def forward(self, predictions, event_indicators, event_times):
        event_likelihood = (
            torch.sum(self._discretize_times(event_times) * predictions[:, :-1], -1)
            + self.tolerance
        )
        nonevent_likelihood = (
            1
            - torch.sum(
                self._get_proportion_of_bins_completed(event_times)
                * predictions[:, :-1],
                -1,
            )
            + self.tolerance
        )

        log_likelihood = event_indicators * torch.log(event_likelihood)
        log_likelihood += (1 - event_indicators) * torch.log(nonevent_likelihood)
        return -1.0 * torch.mean(log_likelihood)
