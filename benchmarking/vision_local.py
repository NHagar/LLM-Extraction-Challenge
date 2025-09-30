import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from tqdm import tqdm

load_dotenv()

# LLM setup - using Mistral vision model
MODEL = "mistralai/magistral-small-2509"

# Configuration
SAMPLE_SIZE = 2  # Set to a number to sample PDFs, None for all
PDF_DIR = Path("deepform/pdfs")

# Load prompts
with open("benchmarking/prompts/baseline_vision.md", "r") as f:
    baseline_prompt = f.read()

with open("benchmarking/prompts/fewshot_vision.md", "r") as f:
    fewshot_prompt = f.read()

# Initialize OpenAI client pointing to LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


# Get all PDF files
pdf_files = sorted(PDF_DIR.glob("*.pdf"))

# Sample PDFs if specified
if SAMPLE_SIZE is not None:
    pdf_files = pdf_files[:SAMPLE_SIZE]
    print(f"Sampled {len(pdf_files)} PDFs for testing")

print(f"Loaded {len(pdf_files)} PDFs from {PDF_DIR}")


def pdf_to_base64_images(pdf_path):
    """Convert PDF pages to base64-encoded images"""
    import base64
    from io import BytesIO

    # Convert PDF to list of PIL images
    images = convert_from_path(pdf_path)

    base64_images = []
    for img in images:
        # Convert PIL image to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        base64_images.append(img_base64)

    return base64_images


def run_inference(model, pdf_path, prompt, prompt_type):
    """Run inference for a single PDF with vision model"""
    max_attempts = 1

    for attempt in range(max_attempts):
        try:
            # Convert PDF pages to base64 images
            images = pdf_to_base64_images(pdf_path)

            # Build message content with all pages as images
            content = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )

            output_text = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            try:
                response_parsed = json.loads(output_text)

                return {
                    "prompt_type": prompt_type,
                    "pdf_name": pdf_path.name,
                    "response": response_parsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except json.JSONDecodeError:
                if attempt < max_attempts - 1:
                    print(
                        f"Parsing error for PDF {pdf_path.name}, retrying... (attempt {attempt + 1}/{max_attempts})"
                    )
                    continue
                else:
                    print(
                        f"Error decoding JSON for PDF {pdf_path.name} with model {model} after {max_attempts} attempts"
                    )
                    print(output_text)
                    return {
                        "prompt_type": prompt_type,
                        "pdf_name": pdf_path.name,
                        "response": {
                            "error": "PARSING_ERROR",
                            "raw_output": output_text,
                        },
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
        except Exception as e:
            if attempt < max_attempts - 1:
                print(
                    f"Error running inference for PDF {pdf_path.name}, retrying... (attempt {attempt + 1}/{max_attempts}): {e}"
                )
                continue
            else:
                print(
                    f"Error running inference for PDF {pdf_path.name} after {max_attempts} attempts: {e}"
                )
                return {
                    "prompt_type": prompt_type,
                    "pdf_name": pdf_path.name,
                    "response": {"error": "INFERENCE_ERROR", "message": str(e)},
                    "input_tokens": 0,
                    "output_tokens": 0,
                }


def run_all_inferences():
    """Run all inferences synchronously"""
    results = []
    prompts = [(baseline_prompt, "baseline"), (fewshot_prompt, "fewshot")]

    print(f"Running inference for model: {MODEL}")
    total_tasks = len(pdf_files) * len(prompts)
    print(f"Total inference tasks: {total_tasks}")

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        for prompt, prompt_type in prompts:
            result = run_inference(MODEL, pdf_path, prompt, prompt_type)
            results.append(result)

    return results


# Run all inferences
inferences = run_all_inferences()

# Write inferences to CSV using pandas with unique timestamp suffix
results_df = pd.DataFrame(inferences)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"benchmarking/data/vision_local_{timestamp}.csv"
results_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Results saved to {output_path}")
