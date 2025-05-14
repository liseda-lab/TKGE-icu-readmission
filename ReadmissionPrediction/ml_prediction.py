import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys 
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from IPython.display import display

#Embedding options: transE_simple, transE_semantic, 
#                   complEx_simple, complEx_semantic
#                   distMult_simple, distMult_semantic
#                   simplE_simple, simplE_semantic

#Model options: LogisticRegression, 
#               RandomForest, 
#               XGBoost

# Define models and hyperparameter grids
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"C": [0.001, 0.01, 0.1]}, #original 0.01,0.1,1,10
        "class_weight": {0: 3, 1: 1} #['balanced'] #Orginal 2:0,67
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [100, 200], 
        "max_depth": [10, 20, None]},
        "min_samples_leaf": [1, 5, 10],
        "class_weight": ['balanced', 'balanced_subsample']
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200], 
            "learning_rate": [0.01, 0.1, 0.2], 
            "scale_pos_weight": [1, 3, 5],  # Add scale_pos_weight for imbalanced classes
            "max_delta_step": [1, 10]       # Add max_delta_step to help with convergence
        }
    }
}


# Get arguments from the command line (ML model)
embedding_name = sys.argv[1]
model_name = sys.argv[2] 

print(f"🔍 Processing {model_name} for {embedding_name}")
# Load data
results = []
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold in range(10):
    X_train = np.load(f"data/model_folds/{embedding_name}/X_train_fold_{fold}.npy")
    X_test = np.load(f"data/model_folds/{embedding_name}/X_test_fold_{fold}.npy")
    y_train = np.load(f"data/model_folds/{embedding_name}/y_train_fold_{fold}.npy")
    y_test = np.load(f"data/model_folds/{embedding_name}/y_test_fold_{fold}.npy")

    model_info = models[model_name]
    print(f"  ▶ Training {model_name} on Fold {fold}")
    start_time = time.time()
    print(f"        ⏳ Running Grid Search")
    grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "fold": fold,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
    
    os.makedirs(f"models/{embedding_name}/{model_name}", exist_ok=True)
    joblib.dump(best_model, f"models/{embedding_name}/{model_name}/best_model_fold_{fold}.pkl")
    
    print(f"    ✅ {model_name} Fold {fold} - AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"    ⏳ Training Time: {time.time() - start_time:.2f} seconds")

# Save results
results_df = pd.DataFrame(results)
# displaying the DataFrame
display(results_df)
# display confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.savefig(f"models/{embedding_name}/{model_name}/confusion_matrix.png", dpi=300, bbox_inches="tight")  # Save as PNG

results_df.to_csv(f"models/{embedding_name}/{model_name}/model_performance.csv", index=False)

print("✅ Model Training Completed!")