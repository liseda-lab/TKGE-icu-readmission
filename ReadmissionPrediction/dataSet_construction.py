import pandas as pd
from datetime import datetime, timedelta
import sys
import os

mimic_dir = sys.argv[1]

patients = pd.read_csv(mimic_dir + "PATIENTS.csv", usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD"])
patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
patients["DOD"] = pd.to_datetime(patients["DOD"], errors="coerce")


icu_stays = pd.read_csv(mimic_dir + "ICUSTAYS.csv", usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"])
icu_stays["INTIME"] = pd.to_datetime(icu_stays["INTIME"], errors="coerce")
icu_stays["OUTTIME"] = pd.to_datetime(icu_stays["OUTTIME"], errors="coerce")

# Load admissions data (contains ETHNICITY and INSURANCE)
admissions = pd.read_csv(mimic_dir + "ADMISSIONS.csv", usecols=["HADM_ID", "ETHNICITY", "INSURANCE"])

# Adjust shifted DOBs for older patients (median imputation)
old_patient = patients['DOB'].dt.year < 2000
date_offset = pd.DateOffset(years=(300-91), days=(-0.4*365))
patients['DOB'][old_patient] = patients['DOB'][old_patient].apply(lambda x: x + date_offset)

# Merge tables
icu_pat = icu_stays.merge(patients, on="SUBJECT_ID", how="left")
# Merge with ethnicity & insurance from admissions
icu_pat = icu_pat.merge(admissions, on="HADM_ID", how="left")
#Handle Missing Data
icu_pat["INSURANCE"] = icu_pat["INSURANCE"].fillna("UNKNOWN")
icu_pat["ETHNICITY"] = icu_pat["ETHNICITY"].fillna("UNKNOWN")
#Fill missing OUTTIME with DOD
icu_pat["OUTTIME"] = icu_pat["OUTTIME"].fillna(icu_pat["DOD"]) #Fill the OUTTIME with DOD
icu_pat = icu_pat.dropna(subset=["OUTTIME"]) #Delete rows with missing OUTTIME

#Exclusion criteria - Under 18, Died inside ICU
icu_pat['AGE'] = (icu_pat['OUTTIME'] - icu_pat['DOB']).dt.days/365.
icu_pat = icu_pat[icu_pat['AGE'] >= 18]
icu_pat = icu_pat[~(icu_pat["DOD"].notna() & (icu_pat["DOD"] == icu_pat["OUTTIME"]))]

# Calculate Length of Stay (LOS in days)
icu_pat["LOS"] = (icu_pat["OUTTIME"] - icu_pat["INTIME"]).dt.total_seconds() / (24 * 3600)  # Convert seconds to days

# Define positive cases (ICU readmission within 30 days or death within 30 days)
icu_pat = icu_pat.sort_values(by=["SUBJECT_ID", "INTIME"])  # Ensure order
icu_pat["DAYS_TO_NEXT"] = icu_pat.groupby("SUBJECT_ID")["INTIME"].shift(-1) - icu_pat["OUTTIME"]
icu_pat["DAYS_TO_NEXT"] = icu_pat["DAYS_TO_NEXT"].dt.days  # Convert to integer days
# Define negative cases - No redmission or death within 30 days or after 30 days
icu_pat["OUTCOME"] = ((icu_pat["DAYS_TO_NEXT"] <= 30) | 
                      (icu_pat["DOD"].notna() & ((icu_pat["DOD"] - icu_pat["OUTTIME"]).dt.days <= 30))).astype(int)

icu_pat = icu_pat[["SUBJECT_ID", "ICUSTAY_ID", "HADM_ID", "GENDER", "ETHNICITY", "INSURANCE", "AGE", "OUTCOME"]]

######################### MINOR ANALYSIS #########################
print("Number of ICU stays:", len(icu_pat))
print("Number of Patients:", len(icu_pat["SUBJECT_ID"].unique()))
print("Number of Positives:", icu_pat["OUTCOME"].sum())
print("Number of Negatives:", (icu_pat["OUTCOME"] == 0).sum())
print("AGE Distribution:\n", icu_pat["AGE"].describe())
#print("LOS Distribuition:\n", icu_pat["LOS"].describe())
icu_counts = icu_pat["SUBJECT_ID"].value_counts()
print("ICU per patient Distribuition\n", icu_counts.describe())  # Check avg, min, and max ICU stays per patient
print("Ethnicity Distribution:\n", icu_pat["ETHNICITY"].value_counts())  
print("Insurance Distribution:\n", icu_pat["INSURANCE"].value_counts())
###################################################################
os.makedirs('data', exist_ok=True)
icu_pat.to_csv('data/dataSet.csv', encoding='utf-8', index=False)