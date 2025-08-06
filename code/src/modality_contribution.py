import json
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from captum.attr import LayerIntegratedGradients

from core.data_utils import MultiModalCollate
from experimental_utils.track_model import select_model
from experimental_utils.load_data import generate_yearly_data
from experimental_utils.load_vocab import load_joint_vocab
from train_clf_asd import get_modality_config

TASK = "raom"
match TASK:
    case "asd":
        TARGET = 8
    case "raom":
        TARGET = 0
    case _:
        raise ValueError
DEVICE = "cuda:0"
EMBEDDING_DIM = 256
N_MODALITY = 4
MODALITY_CONFIG = get_modality_config(['all'])
MODEL_CHECKPOINT_ROOT_PATH = f"../../model_checkpoint/{TASK}-archive"

def load_data():
    # --- load vocab transform ---
    reference_cohort = pd.read_csv(f"../../data/asd/test_asd.csv", dtype={"PATID": str})
    _, transform = load_joint_vocab(reference_cohort)

    # --- load data ---
    test_cohort = pd.read_csv(f"../../data/{TASK}/test_{TASK}.csv")
    test_dataset_list = [
        generate_yearly_data(
            TASK, i, test_cohort, transform, MODALITY_CONFIG, complete_case=False
        )
        for i in tqdm(range(2015, 2023), desc="preparing test")
    ]
    test_dataset = ConcatDataset(test_dataset_list)
    collate_fn = MultiModalCollate(n_modality=N_MODALITY, survival=TASK == "asd")
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    return test_loader


def calculcate_modality_importance(test_loader: DataLoader, layer_ig_module: LayerIntegratedGradients):
    importance = torch.zeros((4,)).to(DEVICE)
    for batch in tqdm(test_loader):
        masks = batch["mask"].to(DEVICE)

        if not masks.all(dim=1).any().item(): # if all samples in the batch do not have complete obs
            continue

        inputs = batch["inputs"].to(DEVICE)
        attr, _ = layer_ig_module.attribute(
            inputs=inputs,
            target=0,
            additional_forward_args=(masks,),
            n_steps=50,
            internal_batch_size=128,
            return_convergence_delta=True
        )
        attr = attr[:, [3,2,0,1], :, :]
        masks = masks[:, [3,2,0,1]]
        attr = attr[masks.all(dim=1)].abs().sum(dim=-1) #(B, 4, L)

        # min-max norm
        attr = (attr - attr.min()) / (attr.max() - attr.min())

        a = attr.sum(dim=-1)
        importance += (a / a.sum(dim=-1, keepdim=True)).mean(dim=0) / len(test_loader)
    return importance


def iqr_masked_mean_std(x: torch.Tensor):
    q1 = x.quantile(0.25, dim=0)
    q3 = x.quantile(0.75, dim=0)
    iqr = q3 - q1

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    within_iqr = (x >= lower) & (x <= upper)  # shape (N, D)
    row_mask = within_iqr.all(dim=1)  # shape (N,)
    x_filtered = x[row_mask]  # shape (N_filtered, D)
    mean = x_filtered.mean(dim=0)
    std = x_filtered.std(dim=0)
    return mean, std


def main():
    test_loader = load_data()

    result = {}

    for modality_checkpoint in ["first_check", "mid_check", "final_check"]:
        for contrast, zp_only in [(False, False), (True, False), (True, True)]:
            filtering_tags = {
                "modality_checkpoint": modality_checkpoint,
                "mixing_approach": "softmax-gating",
                "contrast": contrast,
                "zp_only": zp_only
            }
            importance_list = []
            model_names = select_model(root=MODEL_CHECKPOINT_ROOT_PATH, filtering_tags=filtering_tags)
            for name in model_names:
                print(name)
                # --- load model and set up ig module ---
                model = torch.load(f"{MODEL_CHECKPOINT_ROOT_PATH}/{name}", map_location=DEVICE)
                def forward_func(inputs, masks):
                    return model(inputs, masks)[0]
                lig = LayerIntegratedGradients(forward_func, model.embedding_module)
                
                # --- calculate modality importance ---
                importance = calculcate_modality_importance(test_loader, lig)
                print(importance.round(decimals=3))
                print("")
                
                importance_list.append(importance)

            # --- logging ---
            m = "|".join([filtering_tags["modality_checkpoint"], filtering_tags["mixing_approach"]])
            if filtering_tags["contrast"]:
                m += "|contrast"
            if filtering_tags["zp_only"]:
                m += "|zpOnly"
            mean, std = iqr_masked_mean_std(torch.stack(importance_list))
            result[m] = {'mean': mean.cpu().tolist(), 'std': std.cpu().tolist()}
            with open(f"../../modality_importance/{TASK}.json", "w") as f:
                json.dump(result, f, indent=4)



if __name__ == "__main__":
    main()