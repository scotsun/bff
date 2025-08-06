datatable schema:
    raw:
        for mom's data, there is an extra column for NEWBORN_PATID 
        dx_data: {PATID, ENCID, DX_TYPE, DX, ADMIT_TIME}
        px_data: {PATID, ENCID, PX_TYPE, PX, TIME, ADMIT_DATE}
        med_data: {PATID, ENCID, RXNORM_CUI, RAW_MEDADMIN_MED_NAME, TIME}
        lab_data: {PATID, ENCID, LAB_PX, RAW_LAB_NAME, TIME}
    
    processed:
        {PATID, ENCID,EVENT, TIME, NEWBORN_PATID} for all

