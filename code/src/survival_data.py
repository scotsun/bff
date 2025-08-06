import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import ConcatDataset, DataLoader
from core.data_utils import (
    MultiModalCollate,
    FilteredMultiModalDataset,
)
from train_clf import generate_yearly_data, load_separate_vocab

MODALITY_CONFIG = {
    "birth_newborn": True,
    "developmental": True,
    "birth_mom": True,
    "prenatal": True,
}
MODALITIES = [k for k, v in MODALITY_CONFIG.items() if v]
N_MODALITY = len(MODALITIES)

train_cohort = pd.read_csv("../../data/outcome/train_asd.csv", dtype={"PATID": str})
test_cohort = pd.read_csv("../../data/outcome/test_asd.csv", dtype={"PATID": str})

vocabs, transform = load_separate_vocab(test_cohort)
train_dataset_list = [
    generate_yearly_data(
        i, train_cohort, transform, MODALITY_CONFIG, complete_case=False
    )
    for i in tqdm(range(2015, 2023), desc="preparing train")
]
train_dataset = ConcatDataset(train_dataset_list)
collate_fn = MultiModalCollate(n_modality=N_MODALITY)
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
)

mask_patterns = [[0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
for j, _pattern in enumerate(mask_patterns):
    _data = FilteredMultiModalDataset(train_dataset, _pattern)
    _dataloader = DataLoader(
        _data, batch_size=128, shuffle=False, collate_fn=collate_fn
    )
    events, event_times = [], []
    for batch in tqdm(_dataloader):
        events.append(batch["event"])
        event_times.append(batch["time"])
    events = torch.cat(events)
    event_times = torch.cat(event_times)

    _df = pd.DataFrame({"event": events, "time2event": event_times})
    _df.to_csv(f"../../data/survival/survival_group{j}.csv", index=False)
