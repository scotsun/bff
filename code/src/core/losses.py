import torch
import torch.nn.functional as F
from torch import jit, Tensor


def pairwise_cosine(z: torch.Tensor):
    return F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    """Stable logsumexp."""
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim)
    mask = m == -999

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -999)


def snn_loss(sim: torch.Tensor, pos_target: torch.Tensor, temperature: float):
    n = sim.shape[0]
    sim = sim.clone()
    sim[torch.eye(n).bool()] = -999

    neg_mask = pos_target == 0
    pos = pos_target * sim
    pos[neg_mask] = -999
    loss = -logsumexp(pos / temperature, dim=1) + logsumexp(sim / temperature, dim=1)

    return loss
