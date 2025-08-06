import pandas as pd
from core.data_utils import MultiModalDataGenerator, MultiModalDataset

def generate_yearly_data(
    task: str,
    year: int,
    cohort: pd.DataFrame,
    transform,
    modality_config: dict,
    complete_case: bool = False,
):
    """
    Helper function to generate yearly MultiModalDataset given a birth year.
    """
    dtype = {"PATID": str, "ENCID": str, "NEWBORN_PATID": str}
    yearly_cohort = cohort.loc[cohort["BIRTH_YEAR"] == year]

    modalities = []
    if modality_config["birth_newborn"]:
        birth_newborn = pd.read_csv(
            f"../../data/pretrain/processed/{year}/birth_newborn.csv", dtype=dtype
        )
        birth_newborn = birth_newborn.loc[
            birth_newborn["PATID"].isin(yearly_cohort["PATID"])
        ]
        modalities.append(birth_newborn.groupby("PATID"))
    if modality_config["developmental"]:
        developmental = pd.read_csv(
            f"../../data/pretrain/processed/{year}/developmental.csv", dtype=dtype
        )
        developmental = developmental.loc[
            developmental["PATID"].isin(yearly_cohort["PATID"])
        ]
        modalities.append(developmental.groupby("PATID"))
    if modality_config["birth_mom"]:
        birth_mom = pd.read_csv(
            f"../../data/pretrain/processed/{year}/birth_mom.csv",
            dtype=dtype,
        )
        birth_mom = birth_mom.loc[
            birth_mom["NEWBORN_PATID"].isin(yearly_cohort["PATID"])
        ]
        modalities.append(birth_mom.groupby("NEWBORN_PATID"))
    if modality_config["prenatal"]:
        prenatal = pd.read_csv(
            f"../../data/pretrain/processed/{year}/prenatal.csv",
            dtype=dtype,
        )
        prenatal = prenatal.loc[prenatal["NEWBORN_PATID"].isin(yearly_cohort["PATID"])]
        modalities.append(prenatal.groupby("NEWBORN_PATID"))

    
    match task:
        case "asd":
            outcomes = [
                yearly_cohort["ASD"].values,
                yearly_cohort["Time2ASD_from_Birth"].values,
            ]
        case "raom":
            outcomes = [
                yearly_cohort["rAOM_ever"].values,
            ]
        case _:
            raise ValueError("task not supported")

    gen = MultiModalDataGenerator(
        modalities=modalities,
        patid_list=yearly_cohort["PATID"].values,
        outcomes=outcomes,
        transform=transform,
        max_length=512,
    )
    yearly_data = MultiModalDataset(gen, complete_case=complete_case, n_outcomes=1)

    return yearly_data