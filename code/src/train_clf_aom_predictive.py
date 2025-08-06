import torch
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, random_split, DataLoader
import argparse
import json

from core.trainer import (
    EarlyStopping,
    ForecastAEPreTrainer,
    ForecastDownstreamBinaryTrainer,
)
from core.downstream_models import ForecastingAE
from core.data_utils import tqdm, MultiModalCollate
from train_clf_aom import (
    load_joint_embeddings,
    load_joint_vocab,
    generate_yearly_data,
    get_modality_config,
)


def get_args():
    parser = argparse.ArgumentParser("Train file for AE forecasting approach.")
    parser.add_argument(
        "--modalityCheckpoint",
        type=str,
        help="str: `first_check`, `mid_check`",
        default=None,
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="str: indicate device name."
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    DEVICE = args.device
    MODALITY_CHECKPOINT = args.modalityCheckpoint
    MODALITY_CONFIG = get_modality_config(["all"])

    SEED = np.random.randint(0, 1000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_cohort = pd.read_csv("../../data/raom/train_raom.csv", dtype={"PATID": str})
    test_cohort = pd.read_csv("../../data/raom/test_raom.csv", dtype={"PATID": str})
    reference_cohort = pd.read_csv(
        "../../data/asd/test_asd.csv", dtype={"PATID": str}
    )
    vocab, transform = load_joint_vocab(reference_cohort)
    embeddings = load_joint_embeddings(vocab, 256, DEVICE)

    dataset_list = [
        generate_yearly_data(
            i, train_cohort, transform, MODALITY_CONFIG, complete_case=False
        )
        for i in tqdm(range(2015, 2023), desc="preparing train")
    ]
    total_dataset = ConcatDataset(dataset_list)
    train_dataset, valid_dataset = random_split(
        total_dataset,
        [
            int(0.85 * len(total_dataset)),
            len(total_dataset) - int(0.85 * len(total_dataset)),
        ],
    )
    test_dataset_list = [
        generate_yearly_data(
            i, test_cohort, transform, MODALITY_CONFIG, complete_case=False
        )
        for i in tqdm(range(2015, 2023), desc="preparing test")
    ]
    test_dataset = ConcatDataset(test_dataset_list)
    # -----------------------------------
    # organize data into dataloader
    # -----------------------------------
    collate_fn = MultiModalCollate(n_modality=4, survival=False)
    train_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn
    )
    # -----------------------------------
    # instantiate model and trainer
    # -----------------------------------
    forecasting_ae = ForecastingAE(embedding_module=embeddings, embed_size=256).to(
        DEVICE
    )
    forecasting_ae_trainer = ForecastAEPreTrainer(
        forecasting_ae,
        torch.optim.AdamW(forecasting_ae.parameters(), lr=1e-3),
        early_stopping=None,
        verbose_period=5,
        device=DEVICE,
        modality_selection=MODALITY_CHECKPOINT,
    )
    forecasting_ae_trainer.fit(51, train_loader, valid_loader)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(forecasting_ae.enc_size, forecasting_ae.enc_size),
        torch.nn.ReLU(),
        torch.nn.Linear(forecasting_ae.enc_size, 1),
        torch.nn.Sigmoid(),
    ).to(DEVICE)

    model_path = (
        f"../../model_checkpoint/raom/forecast-{MODALITY_CHECKPOINT}-{SEED}.pth"
    )
    early_stopping = EarlyStopping(patience=5, save_path=model_path, mode="max")
    trainer = ForecastDownstreamBinaryTrainer(
        ae=forecasting_ae,
        model=mlp,
        optimizer=torch.optim.AdamW(mlp.parameters(), lr=1e-3),
        early_stopping=early_stopping,
        criterion=torch.nn.BCELoss(),
        verbose_period=5,
        device=DEVICE,
        modality_selection=MODALITY_CHECKPOINT,
    )
    trainer.fit(31, train_loader, valid_loader)
    # -----------------------------------
    # eval on test dataloader & log
    # -----------------------------------
    test_log = trainer._valid(test_loader, True, 0, bootstrapping=True)
    log_path = f"../../model_result/raom-forecast/{MODALITY_CHECKPOINT}-{SEED}.json"
    with open(log_path, "w") as json_file:
        json.dump(test_log, json_file, indent=4)


if __name__ == "__main__":
    main()
