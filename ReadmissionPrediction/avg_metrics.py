import pandas as pd
import glob

embeddings = [
    #"complEx_semantic",
    #"complEx_simple",
    #"distMult_semantic",
    #"distMult_simple",
    #"rdf2vec_semantic",
    #"rdf2vec_simple",
    #"simplE_semantic",
    #"simplE_simple",
    #"transE_semantic",
    #"transE_simple"
    #"TcomplEx_semantic",
    #"TcomplEx_simple",
    #"TNTcomplEx_semantic",
    #"TNTcomplEx_simple",
    #"TTransE_simple",
    #"TA-distmult_simple"
]

models = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost"
]

merged_summary = pd.DataFrame()

for embedding in embeddings:
    for model in models:
        # Get the file for the specific model and embedding
        file_path = f"models/{embedding}/{model}/model_performance.csv"

        df = pd.read_csv(file_path)

        # Define the metrics to summarize
        metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]

        # Compute mean and std per model
        summary = df.groupby("model")[metrics].agg(["mean", "std"])
        
        # Create a new DataFrame for this model and embedding
        summary["embedding"] = embedding
        summary['model'] = model
        
        # Convert the summary into a DataFrame
        #summary = pd.DataFrame(summary, index=[0])

        # Append the formatted summary to the merged_summary DataFrame
        merged_summary = pd.concat([merged_summary, summary], ignore_index=True)
        print(merged_summary)

# Reorder columns to have 'embedding' at the end
merged_summary = merged_summary[['embedding','model', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']]

# Save the final result to a CSV file
merged_summary.to_csv("models/smmary_results.csv", index=False)

print("Summary saved to summary_results.csv")