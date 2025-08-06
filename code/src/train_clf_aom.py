import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import argparse
from sksurv.util import Surv
from core.data_utils import FilteredMultiModalDataset, MultiModalCollate
from core.downstream_models import (
    BackboneModel,
    DiscreteTimeNN,
    DiscreteFailureTimeNLL,
    MultiModalSNN,
)
from core.trainer import BinaryTrainer, DiscreteTimeNNTrainer, EarlyStopping
from experimental_utils.load_data import generate_yearly_data
from experimental_utils.load_vocab import load_joint_vocab, load_separate_vocab
from experimental_utils.load_embeddings import load_joint_embeddings, load_separate_embeddings


def get_modality_config(flag_list: list[str]) -> dict[str, bool]:
    modality_config = {
        "birth_newborn": False,
        "developmental": False,
        "birth_mom": False,
        "prenatal": False,
    }
    # check for all
    if any([f == "all" for f in flag_list]):
        for k in modality_config.keys():
            modality_config[k] = True
        return modality_config
    # check flag one-by-one
    for flag in flag_list:
        match flag:
            case "birth":
                modality_config["birth_mom"] = True
                modality_config["birth_newborn"] = True
            case "developmental":
                modality_config["developmental"] = True
            case "prenatal":
                modality_config["prenatal"] = True
            case _:
                raise ValueError("Incorrect value for modality flag.")
    return modality_config



def get_args():
    parser = argparse.ArgumentParser(
        description="Train file for downstream bin/surv tasks."
    )
    parser.add_argument(
        "--epoch", type=int, default=51, help="int: number of epochs; default 51"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="float: learning rate; default 1e-4"
    )
    parser.add_argument(
        "--embeddingMode",
        type=str,
        default="joint",
        help="str: `joint` or `separate; default joint`",
    )
    parser.add_argument(
        "--embeddingDim", type=int, default=256, help="int: embedding size; default 256"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        type=str,
        default=["all"],
        help="list[str]: `all`, `birth`, `developmental`, `prenatal`; default [`all`]",
    )
    parser.add_argument(
        "--subgroupAnalysis", action="store_true", help="flag indicator"
    )
    parser.add_argument("--addContrastive", action="store_true", help="flag indicator")
    parser.add_argument(
        "--mixingModule",
        type=str,
        help="str: `softmax-gating`, `self-attention`, or `masked-avg`",
    )
    parser.add_argument(
        "--zpOnly",
        action="store_true",
        help="flag indicator for only using c representation",
    )
    parser.add_argument("--task", type=str, help="str: `binary` or `survival`")

    parser.add_argument(
        "--modalityCheckpoint",
        type=str,
        help="str: `first_check`, `mid_check`, `final_check`",
        default=None,
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="str: indicate device name"
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    EPOCHS = args.epoch
    DEVICE = args.device
    LR = args.lr
    EMBEDDING_DIM = args.embeddingDim
    EMBEDDING_MODE = args.embeddingMode
    MODALITY_FLAGS = args.modalities
    TASK = args.task
    MODALITY_CHECKPOINT = args.modalityCheckpoint

    TRAIN = True
    MODALITY_CONFIG = get_modality_config(MODALITY_FLAGS)
    MODALITIES = [k for k, v in MODALITY_CONFIG.items() if v]
    N_MODALITY = len(MODALITIES)
    BIN_BOUNDARIES = np.arange(1, 10) * 365
    STRATIFIED_ANALYSIS = (N_MODALITY == 4) and args.subgroupAnalysis
    CONTRA_MOD = MultiModalSNN(DEVICE, 0.1) if args.addContrastive else None
    ZP_ONLY = args.zpOnly
    MIXING_MODULE = args.mixingModule

    SEED = np.random.randint(0, 1000)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(DEVICE, "n modality:", N_MODALITY)
    train_cohort = pd.read_csv("../../data/raom/train_raom.csv", dtype={"PATID": str})
    test_cohort = pd.read_csv("../../data/raom/test_raom.csv", dtype={"PATID": str})
    # reference cohort
    reference_cohort = pd.read_csv(
        "../../data/asd/test_asd.csv", dtype={"PATID": str}
    )

    # -----------------------------------
    # load vocab & embeddings
    # -----------------------------------
    match EMBEDDING_MODE:
        case "joint":
            vocab, transform = load_joint_vocab(reference_cohort)
            embeddings = load_joint_embeddings(vocab, EMBEDDING_DIM, DEVICE)
        case "separate":
            vocabs, transform = load_separate_vocab(reference_cohort, MODALITIES)
            embeddings = load_separate_embeddings(
                vocabs, EMBEDDING_DIM, MODALITIES, DEVICE
            )
        case _:
            raise ValueError("embedding_mode only takes `joint` or `separate`")
    # -----------------------------------
    # prepare train, valid, test data
    # -----------------------------------
    dataset_list = [
        generate_yearly_data(
            "raom", i, train_cohort, transform, MODALITY_CONFIG, complete_case=False
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
    collate_fn = MultiModalCollate(n_modality=N_MODALITY, survival=TASK == "survival")
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # -----------------------------------
    # determine downstream task
    # -----------------------------------
    model_tags = [
        f"{EMBEDDING_MODE}",
        f"{TASK}",
        f"{str.join('_', MODALITY_FLAGS)}",
        f"{MODALITY_CHECKPOINT}",
        f"{MIXING_MODULE}",
    ]
    if args.addContrastive:
        model_tags.append("contrast")
    if args.zpOnly:
        model_tags.append("zpOnly")
    model_tags.append(str(SEED))
    model_path = "../../model_checkpoint/raom/" + "_".join(model_tags)

    early_stopping = EarlyStopping(
        patience=5,
        save_path=model_path + ".pth",
        mode="max",
    )

    zp_only = ZP_ONLY

    print("using patient representation only:", zp_only)

    match TASK:
        case "survival":  # discrete time-to-event
            model = DiscreteTimeNN(
                n_bins=len(BIN_BOUNDARIES) - 1,
                embedding_module=embeddings,
                n_modality=N_MODALITY,
                embed_size=EMBEDDING_DIM,
                zp_only=zp_only,
                mixing_module=MIXING_MODULE,
            ).to(DEVICE)
            if CONTRA_MOD:
                parameters = list(model.parameters()) + list(CONTRA_MOD.parameters())
            else:
                parameters = model.parameters()
            optimizer = torch.optim.AdamW(parameters, lr=LR)
            criterion = DiscreteFailureTimeNLL(BIN_BOUNDARIES, device=DEVICE)
            trainer = DiscreteTimeNNTrainer(
                model,
                optimizer,
                early_stopping,
                criterion,
                BIN_BOUNDARIES,
                1,
                DEVICE,
                CONTRA_MOD,
                MODALITY_CHECKPOINT,
            )

        case "binary":  # binary classification
            model = BackboneModel(
                embedding_module=embeddings,
                n_modality=N_MODALITY,
                embed_size=EMBEDDING_DIM,
                zp_only=zp_only,
                mixing_module=MIXING_MODULE,
            ).to(DEVICE)
            if CONTRA_MOD:
                parameters = list(model.parameters()) + list(CONTRA_MOD.parameters())
            else:
                parameters = model.parameters()
            optimizer = torch.optim.AdamW(parameters, lr=LR)
            criterion = nn.BCELoss()
            trainer = BinaryTrainer(
                model,
                optimizer,
                early_stopping,
                criterion,
                1,
                DEVICE,
                CONTRA_MOD,
                MODALITY_CHECKPOINT,
            )
        case _:
            raise ValueError("undefined downstream task")
    if TRAIN:
        # skip training part and directly jump to evaluation if TRAIN is False
        trainer.fit(epochs=EPOCHS, train_loader=train_loader, valid_loader=valid_loader)
        plt.plot(trainer.train_minibatch_loss)
        plt.savefig(model_path + ".png")
    else:
        # collect train outcome for evaluation
        events, times = [], []
        for batch in train_loader:  # iterate thru dataloader & create surv outcome
            event = batch["event"].float().to(DEVICE)
            time = batch["time"].float().to(DEVICE)
            events.append(event)
            times.append(time)
        trainer.train_surv_outcome = Surv.from_arrays(
            event=torch.cat(events).cpu(), time=torch.cat(times).cpu()
        )

    # eval on test dataloader
    trainer.model = torch.load(model_path + ".pth", map_location=DEVICE)
    test_log = trainer._valid(test_loader, True, 0, bootstrapping=True)

    log_path = "../../model_result/raom/" + "_".join(model_tags) + ".json"
    with open(log_path, "w") as json_file:
        json.dump(test_log, json_file, indent=4)

    if CONTRA_MOD:
        print("tau_t", CONTRA_MOD.log_tau_modality.exp())
        print("tau_p", CONTRA_MOD.log_tau_instance.exp())
    # -----------------------------------
    # optional stratified analysis
    # -----------------------------------
    if STRATIFIED_ANALYSIS:
        mask_patterns = [[0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
        for _pattern in mask_patterns:
            _data = FilteredMultiModalDataset(test_dataset, _pattern)
            _dataloader = DataLoader(
                _data, batch_size=128, shuffle=False, collate_fn=collate_fn
            )
            print(f"{_pattern}: {len(_data)}")
            trainer._valid(_dataloader, True, 0)


if __name__ == "__main__":
    main()
