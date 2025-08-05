import json

from ollama import ChatResponse, chat
from sqlite_utils import Database

model = "mistral-small"
model_file = "mistral_small"


def process_email(email):
    try:
        response: ChatResponse = chat(
            model=model,  # Update this to match your installed model name
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": "Produce a JSON object with the following keys: 'committee', which is the name of the committee in the disclaimer that begins with Paid for by but does not include `Paid for by`, the committee address or the treasurer name. If no committee is present, the value of 'committee' should be None. Also add a key called 'sender', which is the name of the person, if any, mentioned as the author of the email. If there is no person named, the value is None. Do not include any other text, no yapping.",
                },
                {"role": "user", "content": email["body"]},
            ],
        )
        # Extract the content from the response
        response_content = response.message.content
        # Parse the JSON response
        parsed_response = json.loads(response_content)
        return parsed_response | email
    except Exception as e:
        print(f"Error processing email {email['id']}: {e}")
        return None


def main(year, month, name):
    db = Database("emails.db")
    entities = []
    failures = []

    for email in db["emails"].rows_where(
        f"year = {year} and month = {month} and disclaimer = 'True'",
        order_by="date",
        limit=1000,
    ):
        print(email["subject"])
        result = process_email(email)
        if result:
            entities.append(result)
        else:
            failures.append(email)

    # Save initial results
    entities_file = f"{model_file}_{name}_{year}.json"
    failures_file = f"{model_file}_{name}_{year}_failures.json"

    with open(entities_file, "w") as file:
        json.dump(entities, file, indent=4)

    with open(failures_file, "w") as file:
        json.dump(failures, file, indent=4)
