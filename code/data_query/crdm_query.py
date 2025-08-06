import pyodbc
import pandas as pd


def cursor2df(cursor: pyodbc.Cursor):
    """Convert records from cursor to dataframe."""
    data_records = cursor.fetchall()
    df = pd.DataFrame.from_records(
        data_records, columns=[col[0] for col in cursor.description]
    )
    return df


def preprocess_data(
    cursor: pyodbc.Cursor, data_category: str, is_mom: bool = False
) -> pd.DataFrame:
    """Preprocess data in to the standardized format."""
    df = cursor2df(cursor)
    match data_category:
        case "dx":
            df["DX"] = df["DX"].str.replace("-", " ")
            df["EVENT"] = df.apply(
                lambda row: "-".join(["dx", str(row["DX_TYPE"]), row["DX"]]),
                axis=1,
            )
            df["TIME"] = df["ADMIT_DATE"].dt.date
        case "px":
            df["PX"] = df["PX"].str.replace("-", " ")
            df["EVENT"] = df.apply(
                lambda row: "-".join(["px", row["PX_TYPE"], row["PX"]]), axis=1
            )
            df["TIME"] = df["TIME"].dt.date
        case "med":
            df["RAW_MEDADMIN_MED_NAME"] = df["RAW_MEDADMIN_MED_NAME"].str.replace(
                "-", " "
            )
            df["EVENT"] = df.apply(
                lambda row: "-".join(["med", row["RAW_MEDADMIN_MED_NAME"]]), axis=1
            )
            df["TIME"] = df["TIME"].dt.date
        case "lab":
            df["LAB_PX"] = df["LAB_PX"].str.replace("-", " ")
            df["EVENT"] = df.apply(lambda row: "-".join(["lab", row["LAB_PX"]]), axis=1)
            df["TIME"] = df["TIME"].dt.date
        case _:
            raise ValueError("data_category must be in {'dx', 'px', 'med', 'lab'}")
    if is_mom:
        return df[["PATID", "ENCID", "EVENT", "TIME", "NEWBORN_PATID"]]
    else:
        return df[["PATID", "ENCID", "EVENT", "TIME"]]


def get_birth_modality(conn: pyodbc.Connection, year: int, is_mom: bool = False):
    patid_str = "PATID, NEWBORN_PATID" if is_mom else "PATID"
    pat = "MOM" if is_mom else "BABY"

    dx_cursor = conn.cursor()
    dx_cursor.execute(
        f"""
        SELECT {patid_str}, dx.ENCOUNTERID as ENCID, DX_TYPE, DX, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_DIAGNOSIS_VW dx
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb 
        ON nb.{pat}_DELIVERY_ENCOUNTERID = dx.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        """
    )
    dx = preprocess_data(dx_cursor, "dx", is_mom)
    dx_cursor.close()
    print("retrieved dx data")

    px_cursor = conn.cursor()
    px_cursor.execute(
        f"""
        SELECT {patid_str}, px.ENCOUNTERID as ENCID, PX_TYPE, PX, PX_DATE as TIME, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_PROCEDURE_VW px
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        ON nb.{pat}_DELIVERY_ENCOUNTERID = px.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        """
    )
    px = preprocess_data(px_cursor, "px", is_mom)
    px_cursor.close()
    print("retrieved px data")

    med_cursor = conn.cursor()
    med_cursor.execute(
        f"""
        SELECT {patid_str}, med.ENCOUNTERID as ENCID, RAW_MEDADMIN_MED_NAME, MEDADMIN_START_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_MED_ADMIN_VW med
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        ON nb.{pat}_DELIVERY_ENCOUNTERID = med.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        """
    )
    med = preprocess_data(med_cursor, "med", is_mom)
    med_cursor.close()
    print("retrieved med data")

    lab_cursor = conn.cursor()
    lab_cursor.execute(
        f"""
        SELECT {patid_str}, lab.ENCOUNTERID as ENCID, LAB_PX, SPECIMEN_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_LAB_VW lab
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        ON nb.{pat}_DELIVERY_ENCOUNTERID = lab.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        """
    )
    lab = preprocess_data(lab_cursor, "lab", is_mom)
    lab_cursor.close()
    print("retrieved lab data")

    return pd.concat([dx, px, med, lab]).sort_values(["PATID", "TIME"])


def get_dev_modality(conn: pyodbc.Connection, year: int):
    dx_cursor = conn.cursor()
    dx_cursor.execute(
        f"""
        SELECT dm.PATID, dx.ENCOUNTERID as ENCID, DX_TYPE, DX, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_DEMOGRAPHIC_VW dm
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_DIAGNOSIS_VW dx
        ON dm.PATID = dx.PATID 
        WHERE EXTRACT(year FROM BIRTH_DATE) = {year}
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) < 730
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) > 0
        """
    )
    dx = preprocess_data(dx_cursor, "dx")
    dx_cursor.close()
    print("retrieved dx data")

    px_cursor = conn.cursor()
    px_cursor.execute(
        f"""
        SELECT dm.PATID, px.ENCOUNTERID as ENCID, PX_TYPE, PX, PX_DATE as TIME, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_DEMOGRAPHIC_VW dm
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_PROCEDURE_VW px
        ON dm.PATID = px.PATID 
        WHERE EXTRACT(year FROM BIRTH_DATE) = {year}
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) < 730
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) > 0
        """
    )
    px = preprocess_data(px_cursor, "px")
    px_cursor.close()
    print("retrieved px data")

    med_cursor = conn.cursor()
    med_cursor.execute(
        f"""
        SELECT med.PATID, med.ENCOUNTERID as ENCID, RAW_MEDADMIN_MED_NAME, MEDADMIN_START_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_MED_ADMIN_VW med
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_DEMOGRAPHIC_VW dm
        ON dm.PATID = med.PATID INNER JOIN
        BI_ACE_VIEWS.PCOR_RSH_ENC_VW enc
        ON med.ENCOUNTERID = enc.ENCOUNTERID
        WHERE EXTRACT(year FROM dm.BIRTH_DATE) = {year}
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) < 730
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) > 0
        """
    )
    med = preprocess_data(med_cursor, "med")
    med_cursor.close()
    print("retrieved med data")

    lab_cursor = conn.cursor()
    lab_cursor.execute(
        f"""
        SELECT lab.PATID, lab.ENCOUNTERID as ENCID, LAB_PX, SPECIMEN_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_LAB_VW lab
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_DEMOGRAPHIC_VW dm
        ON dm.PATID = lab.PATID INNER JOIN
        BI_ACE_VIEWS.PCOR_RSH_ENC_VW enc
        ON lab.ENCOUNTERID = enc.ENCOUNTERID
        WHERE EXTRACT(year FROM dm.BIRTH_DATE) = {year}
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) < 730
        AND TRUNC(ADMIT_DATE) - TRUNC(BIRTH_DATE) > 0
        """
    )
    lab = preprocess_data(lab_cursor, "lab")
    lab_cursor.close()
    print("retrieved lab data")

    return pd.concat([dx, px, med, lab]).sort_values(["PATID", "TIME"])


def get_prenatal_modality(conn: pyodbc.Connection, year: int):
    dx_cursor = conn.cursor()
    dx_cursor.execute(
        f"""
        SELECT PATID, NEWBORN_PATID, dx.ENCOUNTERID as ENCID, DX_TYPE, DX, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_DIAGNOSIS_VW dx
        ON nb.MOM_PATID = dx.PATID 
        WHERE EXTRACT(year FROM DELIVERY_DT) = {year}
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) < 280
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) > 0
        """
    )
    dx = preprocess_data(dx_cursor, "dx", is_mom=True)
    dx_cursor.close()
    print("retrieved dx data")

    px_cursor = conn.cursor()
    px_cursor.execute(
        f"""
        SELECT PATID, NEWBORN_PATID, px.ENCOUNTERID as ENCID, PX_TYPE, PX, PX_DATE as TIME, ADMIT_DATE
        FROM BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_PROCEDURE_VW px
        ON nb.MOM_PATID = px.PATID 
        WHERE EXTRACT(year FROM DELIVERY_DT) = {year}
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) < 280
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) > 0
        """
    )
    px = preprocess_data(px_cursor, "px", is_mom=True)
    px_cursor.close()
    print("retrieved px data")

    med_cursor = conn.cursor()
    med_cursor.execute(
        f"""
        SELECT med.PATID, NEWBORN_PATID, med.ENCOUNTERID as ENCID, RAW_MEDADMIN_MED_NAME, MEDADMIN_START_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_MED_ADMIN_VW med
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        ON nb.MOM_PATID = med.PATID INNER JOIN
        BI_ACE_VIEWS.PCOR_RSH_ENC_VW enc
        ON med.ENCOUNTERID = enc.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) < 280
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) > 0
        """
    )
    med = preprocess_data(med_cursor, "med", is_mom=True)
    med_cursor.close()
    print("retrieved med data")

    lab_cursor = conn.cursor()
    lab_cursor.execute(
        f"""
        SELECT lab.PATID, NEWBORN_PATID, lab.ENCOUNTERID as ENCID, LAB_PX, SPECIMEN_DATE as TIME
        FROM BI_ACE_VIEWS.PCOR_RSH_LAB_VW lab
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_BIRTH_VW nb
        ON nb.MOM_PATID = lab.PATID INNER JOIN
        BI_ACE_VIEWS.PCOR_RSH_ENC_VW enc
        ON lab.ENCOUNTERID = enc.ENCOUNTERID
        WHERE EXTRACT(year FROM nb.DELIVERY_DT) = {year}
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) < 280
        AND TRUNC(DELIVERY_DT) - TRUNC(ADMIT_DATE) > 0
        """
    )
    lab = preprocess_data(lab_cursor, "lab", is_mom=True)
    lab_cursor.close()
    print("retrieved lab data")

    return pd.concat([dx, px, med, lab]).sort_values(["PATID", "TIME"])


# TODO: get other modality generating functions


def preprocess_data_V2(
    cursor: pyodbc.Cursor, data_category: str, cohort_data: pd.DataFrame
) -> pd.DataFrame:
    df = cursor2df(cursor)
    match data_category:
        case "px":
            df["PX"] = df["PX"].str.replace("-", " ")
            df["EVENT"] = df.apply(
                lambda row: "-".join(["px", row["PX_TYPE"], row["PX"]]), axis=1
            )
            df["TIME"] = df["TIME"].dt.date

            df = pd.merge(df, cohort_data[["PATID", "ENC_DATE_D", "ASD"]], on="PATID")
            df = df[
                (df["ASD"] == 0)
                | ((df["ASD"] == 1) & (df["ADMIT_TIME"] <= df["ENC_DATE_D"]))
            ]
            df["t"] = (df["ADMIT_TIME"] <= df["BIRTH_DATE"]) / pd.Timedelta(days=365.25)
        case _:
            raise ValueError("data category note implemented yet")

    return df[["PATID", "ENCID", "EVENT", "TIME", "t", "FLAGWELLCHILD"]]


def get_dev_modality_V2(conn: pyodbc.Connection, year: int):
    px_cursor = conn.cursor()
    px_cursor.execute(
        f"""
        SELECT dm.PATID, TRUNC(dm.BIRTH_DATE) as BIRTH_DATE, px.ENCOUNTERID as ENCID, 
        px.PX_TYPE, px.PX, px.PX_DATE as TIME, TRUNC(px.ADMIT_DATE) as ADMIT_DATE, FLAGWELLCHILD
        FROM BI_ACE_VIEWS.PCOR_RSH_DEMOGRAPHIC_VW dm
        INNER JOIN BI_ACE_VIEWS.PCOR_RSH_PROCEDURE_VW px
        ON dm.PATID = px.PATID INNER JOIN
        BI_ACE_VIEWS.PCOR_RSH_ENC_OP_VW enc
        ON px.ENCOUNTERID = enc.ENCOUNTERID
        WHERE EXTRACT(year FROM dm.BIRTH_DATE) = {year}
        AND TRUNC(px.ADMIT_DATE) - TRUNC(dm.BIRTH_DATE) > 0
        """
    )
    px = preprocess_data_V2(px_cursor, "px")
    px_cursor.close()
    print("retrieved px data")

    return pd.concat([px]).sort_values(["PATID", "TIME"])
