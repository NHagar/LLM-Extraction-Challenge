import csv
import json

from cost import calculate_openai_cost
from dotenv import load_dotenv
from models import Newsletter
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# LLM setup
models = [
    "gpt-4.1-nano-2025-04-14",
]

with open("benchmarking/prompts/baseline.md", "r") as f:
    baseline_prompt = f.read()

with open("benchmarking/prompts/fewshot.md", "r") as f:
    fewshot_prompt = f.read()

client = OpenAI()


# Load training data from CSV and create Newsletter instances
newsletters = []
with open("fundraising-emails/training.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        newsletters.append(Newsletter(**row))

print(f"Loaded {len(newsletters)} newsletters from training data")

inferences = []
# run inference for each model
for model in models:
    print(f"Running inference for model: {model}")
    for newsletter in tqdm(newsletters):
        response = client.responses.create(
            model=model,
            instructions=baseline_prompt,
            input=newsletter.body,
            temperature=0.0,
        )

        cost = calculate_openai_cost(response)

        try:
            response_parsed = json.loads(response.output_text)
            committee_name = response_parsed["committee"]
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON for newsletter {newsletter.uuid} with model {model}"
            )
            committee_name = "<PARSING ERROR>"

        inferences.append(
            {
                "model": model,
                "newsletter_id": newsletter.uuid,
                "committee_name_inferred": committee_name,
                "committee_name_expected": newsletter.committee,
                "cost": cost,
            }
        )

# Write inferences to CSV
with open("benchmarking/inferences.csv", "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "model",
        "newsletter_id",
        "committee_name_inferred",
        "committee_name_expected",
        "cost",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for inference in inferences:
        writer.writerow(inference)
