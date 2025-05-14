import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# Load the dataset
df = pd.read_csv('data/dataSet.csv')

# Get the overall OUTCOME ratio per patient - used for stratification
patient_outcomes = df.groupby("SUBJECT_ID")["OUTCOME"].mean().reset_index()
# Stratify patients based on whether they have **any** positive cases
patient_outcomes["STRATIFY_LABEL"] = (patient_outcomes["OUTCOME"] > 0).astype(int)  # 1 if at least one positive case, else 0
# 10-Fold Stratified Split based on patient-level outcome ratio
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


os.makedirs("data/data_folds", exist_ok=True)
with open("data/data_folds/splitStats.txt", "w") as file:
    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_outcomes, patient_outcomes["STRATIFY_LABEL"])):
        train_patients = patient_outcomes.iloc[train_idx]["SUBJECT_ID"]
        test_patients = patient_outcomes.iloc[test_idx]["SUBJECT_ID"]

        # Assign ICU stays based on the split
        train_df = df[df["SUBJECT_ID"].isin(train_patients)]
        test_df = df[df["SUBJECT_ID"].isin(test_patients)]

        # Save each fold
        train_df.to_csv(f"data/data_folds/train_fold_{fold}.csv", index=False)
        test_df.to_csv(f"data/data_folds/test_fold_{fold}.csv", index=False)

        file.write(f"Fold {fold}\n")
        file.write(f"Train Positives: {train_df['OUTCOME'].sum()} ({train_df['OUTCOME'].mean()*100:.2f}%)\n")
        file.write(f'Train Negatives: {(train_df["OUTCOME"] == 0).sum()} ({(1 - train_df["OUTCOME"].mean())*100:.2f}%)\n')
        file.write(f"Test Positives: {test_df['OUTCOME'].sum()} ({test_df['OUTCOME'].mean()*100:.2f}%)\n")
        file.write(f'Test Negatives: {(test_df["OUTCOME"] == 0).sum()} ({(1 - test_df["OUTCOME"].mean())*100:.2f}%)\n')
        file.write(f"Train Patients: {len(train_patients)}, Test Patients: {len(test_patients)}\n")
        file.write(f"Train ICU stays: {len(train_df)}, Test ICU stays: {len(test_df)}\n")
        file.write("\n")