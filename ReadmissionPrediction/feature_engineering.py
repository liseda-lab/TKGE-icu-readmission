import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from tqdm import tqdm
import os
import sys

def load_embeddings(embedding_file, patient_df):
    """ Load embeddings and merge with patient data based on (subject, hadminid, icu). """
    # Load embeddings file
    embeddings_df = pd.read_csv(embedding_file, sep="\t", header=None)
    # Extract entity
    embeddings_df.rename(columns={0: "entity"}, inplace=True)
    # Dynamically rename embedding columns
    embedding_cols = [f"embedding_{i}" for i in range(1, embeddings_df.shape[1])]  # Assuming first column is 'entity'
    embeddings_df.columns = ["entity"] + embedding_cols

    # Filter df for patients only and filter the URI
    embeddings_df = embeddings_df[embeddings_df["entity"].str.contains("http://purl.obolibrary.org/obo/", na=False)]
    embeddings_df["entity"] = embeddings_df["entity"].str.extract(r"obo/([\d\-]+)$")
    # Extract subject, hadmin, icu from the embeddings' entity column
    embeddings_df[['SUBJECT_ID','HADM_ID','ICUSTAY_ID']] = embeddings_df['entity'].str.split("-", expand=True)
    embeddings_df[['SUBJECT_ID','HADM_ID','ICUSTAY_ID']] = embeddings_df[['SUBJECT_ID','HADM_ID','ICUSTAY_ID']].astype(int)

    # Drop entity column hance it is unecessary and reorder the dataframe
    embeddings_df.drop(columns=['entity'], inplace=True)
    embeddings_df = embeddings_df[["SUBJECT_ID", "ICUSTAY_ID", "HADM_ID"] + \
                        [col for col in embeddings_df.columns if col not in ["SUBJECT_ID", "ICUSTAY_ID", "HADM_ID"]]]
    
    # Merge embeddings with patient data based on (subject, hadminid, icu)
    merged_df = patient_df.merge(embeddings_df, how='inner', on=["SUBJECT_ID", "ICUSTAY_ID", "HADM_ID"])
    ##############################################
    # Merging Stats - Patients lost due to merge #
    ##############################################
    # Calculate the number of patients lost due to merge
    patients_before_merge = patient_df["SUBJECT_ID"].nunique()
    patients_after_merge = merged_df["SUBJECT_ID"].nunique()
    patients_lost = patients_before_merge - patients_after_merge
    print(f"Patients lost due to merge: {patients_lost}")

    return merged_df


def preprocess_features(train_df, test_df, embedding_file):
    """ Prepares features by normalizing age, one-hot encoding categorical variables, and integrating embeddings. """
    features = ["AGE", "GENDER", "INSURANCE", "ETHNICITY"]

    # Load and merge embeddings with the train and test datasets
    print("✅ Loading Train embeddings...")
    train_df = load_embeddings(embedding_file, train_df)
    print("✅ Loading Test embeddings...")
    test_df = load_embeddings(embedding_file, test_df)

    # Separate target variable
    y_train = train_df["OUTCOME"].values
    y_test = test_df["OUTCOME"].values

    # Select feature columns for train and test
    embedding_columns = [f'embedding_{i}' for i in range(1, 301)]  
    features += embedding_columns  # Add embeddings to the feature list
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    print(train_df.shape, X_train.shape, y_train.shape)
    # Normalize continuous variables (Age, LOS)
    scaler = StandardScaler()
    X_train[["AGE"]] = scaler.fit_transform(X_train[["AGE"]])
    X_test[["AGE"]] = scaler.transform(X_test[["AGE"]])

    # One-Hot Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[["GENDER", "INSURANCE", "ETHNICITY"]])
    X_test_cat = encoder.transform(X_test[["GENDER", "INSURANCE", "ETHNICITY"]])

    # Combine numeric + categorical features with embeddings
    X_train_final = np.hstack((X_train[["AGE"]].values, X_train_cat, X_train.drop(columns=["AGE", "GENDER", "INSURANCE", "ETHNICITY"]).values))
    X_test_final = np.hstack((X_test[["AGE"]].values, X_test_cat, X_test.drop(columns=["AGE", "GENDER", "INSURANCE", "ETHNICITY"]).values))

    assert X_train_final.shape[1] == 349, f"Expected 349 features on train, got {X_train_final.shape[1]}"
    assert X_test_final.shape[1] == 349, f"Expected 349 features on test, got {X_test_final.shape[1]}"
    assert y_train.shape[0] == X_train_final.shape[0], f"Train features and target size mismatch"
    assert y_test.shape[0] == X_test_final.shape[0], f"Test features and target size mismatch"

    return X_train_final, X_test_final, y_train, y_test, scaler, encoder


# Load train-test splits (Loop through folds)
embedding_file = sys.argv[1]

# Extract embedding name for saving Based on our naming strucutre
# Model_data_method_dimenions_embeddings.txt
embedding_name = f'{os.path.basename(embedding_file).split("_")[0]}_{os.path.basename(embedding_file).split("_")[2]}'

for fold in tqdm(range(10), desc="📂 Processing Folds"):
    train_df = pd.read_csv(f"data/data_folds/train_fold_{fold}.csv")
    test_df = pd.read_csv(f"data/data_folds/test_fold_{fold}.csv")


    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_features(train_df, test_df, embedding_file)
    print(train_df.shape, X_train.shape, y_train.shape)
    # Save features for each fold
    os.makedirs(f"data/model_folds/{embedding_name}", exist_ok=True)
    np.save(f"data/model_folds/{embedding_name}/X_train_fold_{fold}.npy", X_train)
    np.save(f"data/model_folds/{embedding_name}/X_test_fold_{fold}.npy", X_test)
    np.save(f"data/model_folds/{embedding_name}/y_train_fold_{fold}.npy", y_train)
    np.save(f"data/model_folds/{embedding_name}/y_test_fold_{fold}.npy", y_test)

    # Save preprocessors (important for deployment)
    os.makedirs(f"models/{embedding_name}", exist_ok=True)
    joblib.dump(scaler, f"models/{embedding_name}/scaler_fold_{fold}.pkl")
    joblib.dump(encoder, f"models/{embedding_name}/encoder_fold_{fold}.pkl")

print("✅ Feature Engineering Completed for all folds.")

