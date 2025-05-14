import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from tqdm import tqdm
import os
import sys

def preprocess_features(train_df, test_df):
    """ Prepares features by normalizing age and one-hot encoding categorical variables. """
    features = ["AGE", "GENDER", "INSURANCE", "ETHNICITY"]

    # Separate target variable
    y_train = train_df["OUTCOME"].values
    y_test = test_df["OUTCOME"].values

    # Select feature columns for train and test
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    print(f"🚀 Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # Normalize continuous variables (Age)
    scaler = StandardScaler()
    X_train[["AGE"]] = scaler.fit_transform(X_train[["AGE"]])
    X_test[["AGE"]] = scaler.transform(X_test[["AGE"]])

    # One-Hot Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[["GENDER", "INSURANCE", "ETHNICITY"]])
    X_test_cat = encoder.transform(X_test[["GENDER", "INSURANCE", "ETHNICITY"]])

    # Combine numeric + categorical features (no embeddings)
    X_train_final = np.hstack((X_train[["AGE"]].values, X_train_cat))
    X_test_final = np.hstack((X_test[["AGE"]].values, X_test_cat))

    print(f"🚀 Feature Engineering Completed: {X_train_final.shape[1]} Features")
    assert y_train.shape[0] == X_train_final.shape[0], f"Train features and target size mismatch"
    assert y_test.shape[0] == X_test_final.shape[0], f"Test features and target size mismatch"

    return X_train_final, X_test_final, y_train, y_test, scaler, encoder


# Load train-test splits (Loop through folds)
for fold in tqdm(range(10), desc="📂 Processing Folds"):
    train_df = pd.read_csv(f"data/data_folds/train_fold_{fold}.csv")
    test_df = pd.read_csv(f"data/data_folds/test_fold_{fold}.csv")

    # Process features without embeddings
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_features(train_df, test_df)
    print(f"✅ Fold {fold} | Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    # Save features for each fold
    os.makedirs(f"data/model_folds/no_embeddings", exist_ok=True)
    np.save(f"data/model_folds/no_embeddings/X_train_fold_{fold}.npy", X_train)
    np.save(f"data/model_folds/no_embeddings/X_test_fold_{fold}.npy", X_test)
    np.save(f"data/model_folds/no_embeddings/y_train_fold_{fold}.npy", y_train)
    np.save(f"data/model_folds/no_embeddings/y_test_fold_{fold}.npy", y_test)

    # Save preprocessors (important for deployment)
    os.makedirs(f"models/no_embeddings", exist_ok=True)
    joblib.dump(scaler, f"models/no_embeddings/scaler_fold_{fold}.pkl")
    joblib.dump(encoder, f"models/no_embeddings/encoder_fold_{fold}.pkl")

print("✅ Feature Engineering Completed for all folds without embeddings.")