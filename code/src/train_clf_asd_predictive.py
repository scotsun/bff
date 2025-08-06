import torch
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, random_split, DataLoader
import argparse
import json

from core.trainer import (
    EarlyStopping,
    ForecastAEPreTrainer,
    ForecastDownstreamTrainer,
    DiscreteTimeNNTrainer,
)
from core.downstream_models import (
    ForecastingAE,
    SimpleTimeNN,
    DiscreteTimeNN,
    MultiModalSNN,
    DiscreteFailureTimeNLL,
)
from core.data_utils import tqdm, MultiModalCollate
from train_clf_asd import (
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
        "--data_usage", type=float, help="float: percent of train", default=0.85
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="str: indicate device name."
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    DATA_USAGE = args.data_usage
    DEVICE = args.device
    MODALITY_CHECKPOINT = args.modalityCheckpoint
    BIN_BOUNDARIES = np.arange(1, 10) * 365
    MODALITY_CONFIG = get_modality_config(["all"])

    SEED = np.random.randint(0, 1000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    tags = [MODALITY_CHECKPOINT, f"p{int(DATA_USAGE * 100)}", str(SEED)]

    train_cohort = pd.read_csv("../../data/asd/train_asd.csv", dtype={"PATID": str})
    test_cohort = pd.read_csv("../../data/asd/test_asd.csv", dtype={"PATID": str})
    reference_cohort = test_cohort = pd.read_csv("../../data/asd/test_asd.csv", dtype={"PATID": str})
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
            int(DATA_USAGE * len(total_dataset)),
            len(total_dataset) - int(DATA_USAGE * len(total_dataset)),
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
    collate_fn = MultiModalCollate(n_modality=4, survival=True)
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
    # instantiate models and trainers
    # -----------------------------------
    ##  forecast ae approach
    forecasting_ae = ForecastingAE(embedding_module=embeddings, embed_size=256).to(
        DEVICE
    )
    forecast_ae_trainer = ForecastAEPreTrainer(
        forecasting_ae,
        torch.optim.AdamW(forecasting_ae.parameters(), lr=1e-3),
        early_stopping=None,
        verbose_period=5,
        device=DEVICE,
        modality_selection=MODALITY_CHECKPOINT,
    )
    forecast_ae_trainer.fit(31, train_loader, valid_loader)

    forecast_early_stopping = EarlyStopping(
        patience=5,
        save_path="../../model_checkpoint/asd-forecast/forecast-"
        + "-".join(tags)
        + ".pth",
        mode="max",
    )
    dtnn = SimpleTimeNN(enc_size=128, n_bins=len(BIN_BOUNDARIES) - 1).to(DEVICE)
    forecast_trainer = ForecastDownstreamTrainer(
        ae=forecasting_ae,
        model=dtnn,
        optimizer=torch.optim.AdamW(dtnn.parameters(), lr=1e-3),
        early_stopping=forecast_early_stopping,
        criterion=DiscreteFailureTimeNLL(BIN_BOUNDARIES, device=DEVICE),
        bin_boundaries=BIN_BOUNDARIES,
        verbose_period=5,
        device=DEVICE,
        modality_selection=MODALITY_CHECKPOINT,
    )
    forecast_trainer.fit(51, train_loader, valid_loader)

    ##  bff
    bff_dtnn = DiscreteTimeNN(
        n_bins=len(BIN_BOUNDARIES) - 1,
        embedding_module=embeddings,
        n_modality=4,
        embed_size=256,
        zp_only=True,
        mixing_module="softmax-gating",
        enc_size=256,  # zp only takes 128 so that same dtnn head is used
    ).to(DEVICE)
    contrastive_module = MultiModalSNN(DEVICE)
    bff_early_stopping = EarlyStopping(
        patience=5,
        save_path="../../model_checkpoint/asd-forecast/bff-" + "-".join(tags) + ".pth",
        mode="max",
    )
    bff_trainer = DiscreteTimeNNTrainer(
        model=bff_dtnn,
        optimizer=torch.optim.AdamW(
            list(bff_dtnn.parameters()) + list(contrastive_module.parameters()), lr=5e-4
        ),
        early_stopping=bff_early_stopping,
        criterion=DiscreteFailureTimeNLL(BIN_BOUNDARIES, device=DEVICE),
        bin_boundaries=BIN_BOUNDARIES,
        verbose_period=1,
        device=DEVICE,
        contrastive_module=contrastive_module,
        modality_selection=MODALITY_CHECKPOINT,
    )
    bff_trainer.fit(51, train_loader, valid_loader)

    # -----------------------------------
    # eval on test dataloader & log
    # -----------------------------------
    forecast_log = forecast_trainer._valid(test_loader, True, 0, bootstrapping=False)
    bff_log = bff_trainer._valid(test_loader, True, 0, bootstrapping=False)
    log_path = "../../model_result/asd-forecast/" + "-".join(tags) + ".json"
    with open(log_path, "w") as json_file:
        log = {"forecast": forecast_log, "bff": bff_log}
        json.dump(log, json_file, indent=4)


if __name__ == "__main__":
    main()
