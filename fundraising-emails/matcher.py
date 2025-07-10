import os
import json
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load the CSV file
csv_file = "training.csv"
df_csv = pd.read_csv(csv_file)

# Directory containing JSON files
json_directory = "."
model_scores = []

# Columns to merge on
merge_columns = ["email", "subject", "year", "month", "day", "hour", "minute", "domain"]

# Initialize a list to store summary data
summary_data = []

# Make sure the evals directory exists
os.makedirs("evals", exist_ok=True)

# Process each JSON file in the directory
for json_filename in os.listdir(json_directory):
    if json_filename.endswith(".json"):
        json_file_path = os.path.join(json_directory, json_filename)
        
        # Load the JSON file
        with open(json_file_path, "r") as f:
            data_json = json.load(f)
        df_json = pd.DataFrame(data_json)
        
        # Ensure consistent data types and clean data
        for col in merge_columns:
            if col in df_csv.columns and col in df_json.columns:
                df_csv[col] = df_csv[col].astype(str).str.strip().str.lower()
                df_json[col] = df_json[col].astype(str).str.strip().str.lower()
        
        # Merge CSV and JSON data on specified attributes
        merged = pd.merge(df_csv, df_json, on=merge_columns, how="inner", suffixes=("_csv", "_json"))
        
        # Ensure the suffixed `committee` columns exist before comparison
        if "committee_csv" in merged.columns and "committee_json" in merged.columns:
            y_true = merged["committee_csv"].fillna("none").astype(str).str.lower().str.strip()
            y_pred = merged["committee_json"].fillna("none").astype(str).str.lower().str.strip()
            merged["committee_match"] = y_true == y_pred
        else:
            merged["committee_match"] = False
        
        # Calculate the statistics
        num_records = len(merged)
        num_matches = merged["committee_match"].sum()
        y_true = merged["committee_csv"].fillna("none").astype(str).str.lower()
        y_pred = merged["committee_json"].fillna("none").astype(str).str.lower()
        precision, recall, f1, *_ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Append the results to the summary data
        summary_data.append({
            "JSON Filename": json_filename,
            "Total Records": num_records,
            "Committee Matches": int((y_true == y_pred).sum()),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
        model_scores.append({
            "JSON Filename": json_filename,
            "Total Records": len(merged),
            "Committee Matches": int((y_true == y_pred).sum()),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
        # Generate the classification report
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        
        # Save as JSON file in the same directory
        json_report_filename = f"evals/classification_report_{json_filename.replace('.json', '')}.json"
        with open(json_report_filename, "w") as f:
            json.dump(report, f, indent=4)
        

# Create a summary DataFrame and save to a CSV file
summary_df = pd.DataFrame(summary_data)
output_file = "summary_all_json.csv"
summary_df.to_csv(output_file, index=False)
print(f"Summary saved to {output_file}")

scores_df = pd.DataFrame(model_scores)
scores_df.to_csv("evals/model_performance_summary.csv", index=False)