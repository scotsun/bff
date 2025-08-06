import pyodbc
import pandas as pd
from crdm_query import get_birth_modality, get_dev_modality, get_prenatal_modality


def remove_birth_enc(df: pd.DataFrame, birth_df: pd.DataFrame):
    """Remove birth encounter from dev/prenatal period."""
    pat_enc = set(zip(birth_df["PATID"], birth_df["ENCID"]))
    df_ = df[
        ~df.apply(
            lambda x: (x["PATID"], x["ENCID"]) in pat_enc,
            axis=1,
        )
    ]
    return df_


def main():
    conn = pyodbc.connect("DSN=DSR_PROD")
    for year in range(2015, 2023):
        print(f"pulling data from {year}")
        df_birth_baby = get_birth_modality(conn=conn, year=year)
        df_birth_baby.to_csv(
            f"./data/pretrain/processed/{year}/birth_newborn.csv", index=False
        )

        df_birth_mom = get_birth_modality(conn=conn, year=year, is_mom=True)
        df_birth_mom.to_csv(
            f"./data/pretrain/processed/{year}/birth_mom.csv", index=False
        )

        df_dev = get_dev_modality(conn=conn, year=year)
        df_dev = remove_birth_enc(df_dev, df_birth_baby)
        df_dev.to_csv(
            f"./data/pretrain/processed/{year}/developmental.csv", index=False
        )

        df_mom = get_prenatal_modality(conn=conn, year=year)
        df_mom = remove_birth_enc(df_mom, df_birth_mom)
        df_mom.to_csv(f"./data/pretrain/processed/{year}/prenatal.csv", index=False)


if __name__ == "__main__":
    main()
