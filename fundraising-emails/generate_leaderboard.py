#!/usr/bin/env python3
import os
import csv
import pandas as pd
from datetime import datetime

def generate_leaderboard_html(summary_csv_path, output_path):
    """
    Generate an HTML leaderboard from the summary CSV file, 
    with sections for Updated Prompt and Original Prompt.
    
    :param summary_csv_path: Path to the summary CSV file
    :param output_path: Path to save the output HTML file
    """
    # Read the summary CSV
    try:
        df = pd.read_csv(summary_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Separate dataframes for updated and original prompts
    df_updated = df[df['JSON Filename'].str.contains('prompt2', case=False)].copy()
    df_original = df[~df['JSON Filename'].str.contains('prompt2', case=False)].copy()

    # Sort both dataframes
    df_updated = df_updated.sort_values(by=['Total Records', 'Committee Matches'], ascending=[False, False])
    df_original = df_original.sort_values(by=['Total Records', 'Committee Matches'], ascending=[False, False])

    # Function to generate table rows
    def generate_table_rows(dataframe):
        return "".join([f"""
            <tr>
                <td>{row['JSON Filename']}</td>
                <td>{row['Total Records']}</td>
                <td>{row['Committee Matches']}</td>
                <td>{row['Committee Matches'] / row['Total Records'] * 100:.2f}%</td>
            </tr>""" for _, row in dataframe.iterrows()])

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Political Email Extraction Leaderboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2 {{
            text-align: center;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Political Email Extraction Leaderboard</h1>
    
    <h2>Updated Prompt</h2>
    <table>
        <thead>
            <tr>
                <th>Model (JSON Filename)</th>
                <th>Total Records</th>
                <th>Committee Matches</th>
                <th>Match Percentage</th>
            </tr>
        </thead>
        <tbody>
            {generate_table_rows(df_updated)}
        </tbody>
    </table>
    
    <h2>Original Prompt</h2>
    <table>
        <thead>
            <tr>
                <th>Model (JSON Filename)</th>
                <th>Total Records</th>
                <th>Committee Matches</th>
                <th>Match Percentage</th>
            </tr>
        </thead>
        <tbody>
            {generate_table_rows(df_original)}
        </tbody>
    </table>
    
    <div class="timestamp">
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>"""

    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Leaderboard HTML generated at {output_path}")

if __name__ == "__main__":
    # Default paths, can be modified as needed
    summary_csv_path = "summary_all_json.csv"
    output_html_path = "index.html"
    
    generate_leaderboard_html(summary_csv_path, output_html_path)