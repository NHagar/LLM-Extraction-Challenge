def calculate_openai_cost(response):
    """
    Calculate the total cost of an OpenAI API request from the response object.

    Args:
        response: OpenAI response object with usage information

    Returns:
        dict: Contains model, input_tokens, output_tokens, input_cost, output_cost, total_cost
    """
    # OpenAI pricing per 1M tokens (as of recent pricing)
    MODEL_PRICING = {
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
        "gpt-4.5-preview": {"input": 75.00, "output": 150.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o3": {"input": 2.00, "output": 8.00},
        "o4-mini": {"input": 1.10, "output": 4.40},
    }

    # Extract information from response
    model = response.model
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens

    # Get pricing for the model
    if model not in MODEL_PRICING:
        raise ValueError(f"Pricing not available for model: {model}")

    pricing = MODEL_PRICING[model]

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
