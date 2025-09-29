import asyncio
import json
from datetime import datetime

import pandas as pd
from cost import calculate_openai_cost
from dotenv import load_dotenv
from models import EmailResponse, Newsletter
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

load_dotenv()

# LLM setup
models = [
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-5-nano-2025-08-07",
]

with open("benchmarking/prompts/baseline.md", "r") as f:
    baseline_prompt = f.read()

with open("benchmarking/prompts/fewshot.md", "r") as f:
    fewshot_prompt = f.read()

client = AsyncOpenAI()


# Load training data from CSV and create Newsletter instances
df = pd.read_csv("fundraising-emails/training.csv", encoding="utf-8")
df = df.fillna("")  # Replace NaN values with empty strings
newsletters = [Newsletter(**row) for _, row in df.iterrows()]

print(f"Loaded {len(newsletters)} newsletters from training data")


async def run_inference(model, newsletter, prompt, prompt_type, semaphore):
    """Run inference for a single newsletter with a given model and prompt"""
    async with semaphore:
        response = await client.responses.create(
            model=model,
            instructions=prompt,
            input=newsletter.body,
            # temperature=0.0,
        )

    cost = calculate_openai_cost(response)

    try:
        response_parsed = json.loads(response.output_text)
        committee_name = response_parsed["committee"]
    except json.JSONDecodeError:
        print(
            f"Error decoding JSON for newsletter {newsletter.uuid} with model {model} and prompt {prompt_type}"
        )
        print(response.output_text)
        committee_name = "<PARSING ERROR>"

    return {
        "prompt_type": prompt_type,
        "newsletter_id": newsletter.uuid,
        "committee_name_inferred": committee_name,
        "committee_name_expected": newsletter.committee,
        **cost,
    }


async def run_inference_structured_output(
    model, newsletter, prompt, prompt_type, semaphore
):
    """Run inference for a single newsletter with a given model and prompt, expecting structured output"""
    async with semaphore:
        response = await client.responses.parse(
            model=model,
            instructions=prompt,
            input=newsletter.body,
            # temperature=0.0,
            text_format=EmailResponse,
        )
    cost = calculate_openai_cost(response)

    try:
        response_parsed = response.output_parsed
        committee_name = response_parsed.committee
    except json.JSONDecodeError:
        print(
            f"Error decoding JSON for newsletter {newsletter.uuid} with model {model} and prompt {prompt_type}"
        )
        print(response.output_parsed)
        committee_name = "<PARSING ERROR>"

    return {
        "prompt_type": f"{prompt_type}_structured",
        "newsletter_id": newsletter.uuid,
        "committee_name_inferred": committee_name,
        "committee_name_expected": newsletter.committee,
        **cost,
    }


async def run_all_inferences():
    """Run all inferences in parallel with rate limiting"""
    # Limit concurrent requests to avoid rate limits
    semaphore = asyncio.Semaphore(20)
    tasks = []
    prompts = [(baseline_prompt, "baseline"), (fewshot_prompt, "fewshot")]

    for model in models:
        print(f"Preparing inference tasks for model: {model}")
        for newsletter in newsletters:
            for prompt, prompt_type in prompts:
                # Add regular inference task
                tasks.append(
                    run_inference(model, newsletter, prompt, prompt_type, semaphore)
                )
                # Add structured output inference task
                tasks.append(
                    run_inference_structured_output(
                        model, newsletter, prompt, prompt_type, semaphore
                    )
                )

    print(f"Running {len(tasks)} inference tasks in parallel (max 20 concurrent)...")
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await task
        results.append(result)

    return results


# Run all inferences in parallel
inferences = asyncio.run(run_all_inferences())

# Write inferences to CSV using pandas with unique timestamp suffix
results_df = pd.DataFrame(inferences)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"benchmarking/data/inferences_{timestamp}.csv"
results_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Results saved to {output_path}")
