import json
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from models import Newsletter
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# LLM setup - using local model via LM Studio
MODEL = "openai/gpt-oss-20b"

# Configuration
SAMPLE_SIZE = 10  # Set to a number to sample newsletters, None for all

with open("benchmarking/prompts/baseline.md", "r") as f:
    baseline_prompt = f.read()

with open("benchmarking/prompts/fewshot.md", "r") as f:
    fewshot_prompt = f.read()

# Initialize OpenAI client pointing to LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


# Load training data from CSV and create Newsletter instances
df = pd.read_csv("fundraising-emails/training.csv", encoding="utf-8")
df = df.fillna("")  # Replace NaN values with empty strings

# Sample newsletters if specified
if SAMPLE_SIZE is not None:
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    print(f"Sampled {len(df)} newsletters for testing")

newsletters = [Newsletter(**row) for _, row in df.iterrows()]
print(f"Loaded {len(newsletters)} newsletters from training data")


def run_inference(model, newsletter, prompt, prompt_type):
    """Run inference for a single newsletter with a given model and prompt"""
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": newsletter.body},
                ],
                temperature=0.0,
            )

            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            try:
                response_parsed = json.loads(output_text)
                committee_name = response_parsed["committee"]

                return {
                    "prompt_type": prompt_type,
                    "newsletter_id": newsletter.uuid,
                    "committee_name_inferred": committee_name,
                    "committee_name_expected": newsletter.committee,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except json.JSONDecodeError:
                if attempt < max_attempts - 1:
                    print(
                        f"Parsing error for newsletter {newsletter.uuid}, retrying... (attempt {attempt + 1}/{max_attempts})"
                    )
                    continue
                else:
                    print(
                        f"Error decoding JSON for newsletter {newsletter.uuid} with model {model} and prompt {prompt_type} after {max_attempts} attempts"
                    )
                    print(output_text)
                    return {
                        "prompt_type": prompt_type,
                        "newsletter_id": newsletter.uuid,
                        "committee_name_inferred": "<PARSING ERROR>",
                        "committee_name_expected": newsletter.committee,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"Error running inference for newsletter {newsletter.uuid}, retrying... (attempt {attempt + 1}/{max_attempts}): {e}")
                continue
            else:
                print(f"Error running inference for newsletter {newsletter.uuid} after {max_attempts} attempts: {e}")
                return {
                    "prompt_type": prompt_type,
                    "newsletter_id": newsletter.uuid,
                    "committee_name_inferred": "<ERROR>",
                    "committee_name_expected": newsletter.committee,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }


def run_all_inferences():
    """Run all inferences synchronously"""
    results = []
    prompts = [(baseline_prompt, "baseline"), (fewshot_prompt, "fewshot")]

    print(f"Running inference for model: {MODEL}")
    total_tasks = len(newsletters) * len(prompts)
    print(f"Total inference tasks: {total_tasks}")

    for newsletter in tqdm(newsletters, desc="Processing newsletters"):
        for prompt, prompt_type in prompts:
            result = run_inference(MODEL, newsletter, prompt, prompt_type)
            results.append(result)

    return results


# Run all inferences
inferences = run_all_inferences()

# Write inferences to CSV using pandas with unique timestamp suffix
results_df = pd.DataFrame(inferences)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"benchmarking/data/inferences_local_{timestamp}.csv"
results_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Results saved to {output_path}")
