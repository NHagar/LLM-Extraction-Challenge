import glob
import json
import os

import pandas as pd


def extract_model_name(filename):
    """
    Extract model name from filename like 'claude37_sonnet_november_2024_prompt2.json'
    Returns 'claude37_sonnet'
    """
    basename = os.path.basename(filename)
    # Remove the .json extension
    name_without_ext = basename.replace(".json", "")
    # Split by underscore and take first two parts
    parts = name_without_ext.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return parts[0]


def load_training_csv(csv_path="training.csv"):
    """
    Load the training CSV file as the base for the combined dataframe
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded training CSV with {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Training CSV not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading training CSV: {e}")
        return None


def load_json_files(directory="."):
    """
    Load all JSON files ending in _prompt2.json from the specified directory
    """
    pattern = os.path.join(directory, "*_prompt2.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return {}

    data = {}
    for file_path in json_files:
        model_name = extract_model_name(file_path)
        print(f"Processing {file_path} -> {model_name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                data[model_name] = json_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    return data


def create_combined_dataframe(training_df, json_data):
    """
    Create a combined dataframe with training data as base and model predictions
    """
    if training_df is None or training_df.empty:
        print("No training data to process")
        return pd.DataFrame()

    # Define the specific columns we want to keep from training data
    keep_columns = [
        "name",
        "email",
        "subject",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "domain",
        "party",
        "disclaimer",
    ]

    # Create base dataframe with selected columns
    df = training_df[keep_columns].copy()

    # Add the canonical committee column (uppercase)
    df["committee"] = training_df["committee"].apply(
        lambda x: str(x).upper() if pd.notna(x) else None
    )

    # Add model predictions
    for model_name, emails in json_data.items():
        print(f"Processing model: {model_name} with {len(emails)} emails")

        # Create a mapping from email attributes to committee prediction
        committee_mapping = {}
        matches_found = 0

        for email in emails:
            # Try multiple matching strategies
            keys_to_try = [
                (
                    email.get("subject", ""),
                    email.get("date", ""),
                    email.get("name", ""),
                ),
                (
                    email.get("subject", ""),
                    email.get("date", ""),
                    email.get("email", ""),
                ),
                (
                    email.get("subject", ""),
                    str(email.get("year", "")),
                    str(email.get("month", "")),
                    str(email.get("day", "")),
                ),
            ]

            committee_value = email.get("committee", None)
            if committee_value is not None:
                committee_value = str(committee_value).upper()

            for key in keys_to_try:
                if key not in committee_mapping:
                    committee_mapping[key] = committee_value

        # Create the column with default None values
        df[model_name] = None

        # Try to match each row in the training data
        for idx, row in df.iterrows():
            keys_to_try = [
                (row.get("subject", ""), row.get("date", ""), row.get("name", "")),
                (row.get("subject", ""), row.get("date", ""), row.get("email", "")),
                (
                    row.get("subject", ""),
                    str(row.get("year", "")),
                    str(row.get("month", "")),
                    str(row.get("day", "")),
                ),
            ]

            for key in keys_to_try:
                if key in committee_mapping:
                    df.at[idx, model_name] = committee_mapping[key]
                    matches_found += 1
                    break

        print(f"  Found {matches_found} matches for {model_name}")

    # Add a count column for non-None model predictions
    model_columns = [
        col for col in df.columns if col not in keep_columns + ["committee"]
    ]
    df["model_count"] = df[model_columns].notna().sum(axis=1)

    # Add a count column for exact matches to the canonical committee value
    def count_matches(row):
        canonical = row["committee"]
        if pd.isna(canonical):
            return 0
        return sum(1 for col in model_columns if row[col] == canonical)

    df["exact_matches"] = df.apply(count_matches, axis=1)

    return df


def main():
    """
    Main function to process training CSV and JSON files to create combined CSV
    """
    print("Starting training CSV and JSON combination process...")

    # Load training CSV
    training_df = load_training_csv()
    if training_df is None:
        return

    # Load all JSON files
    json_data = load_json_files()

    if not json_data:
        print("No JSON files found. Creating CSV with just training data.")
    else:
        print(f"Found {len(json_data)} JSON files to process")

    # Create combined dataframe
    df = create_combined_dataframe(training_df, json_data)

    if df.empty:
        print("No data to write. Exiting.")
        return

    # Write to CSV
    output_file = "combined_matrix.csv"
    df.to_csv(output_file, index=False)

    print(f"Combined dataframe written to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Display first few rows for verification
    print("\nFirst 3 rows:")
    print(df.head(3))


if __name__ == "__main__":
    main()
