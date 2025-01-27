import pandas as pd
import json
import os

# Load the CSV file
csv_file = "training.csv"
df_csv = pd.read_csv(csv_file)

# Directory containing JSON files
json_directory = "."

# Columns to merge on
merge_columns = ["email", "subject", "year", "month", "day", "hour", "minute", "domain"]

# Initialize a list to store summary data
summary_data = []

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
            # Check if the `committee` attributes match
            merged["committee_match"] = merged["committee_csv"] == merged["committee_json"]
        else:
            # If columns do not exist, default to no matches
            merged["committee_match"] = False

        # Calculate the statistics
        num_records = len(merged)
        num_matches = merged["committee_match"].sum()


        # Append the results to the summary data
        summary_data.append({
            "JSON Filename": json_filename,
            "Total Records": num_records,
            "Committee Matches": num_matches
        })

# Create a summary DataFrame and save to a CSV file
summary_df = pd.DataFrame(summary_data)
output_file = "summary_all_json.csv"
summary_df.to_csv(output_file, index=False)

print(f"Summary saved to {output_file}")
