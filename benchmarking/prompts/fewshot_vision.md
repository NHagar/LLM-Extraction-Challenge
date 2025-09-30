Produce a JSON object from this invoice with the keys:
'gross_amount': total amount of invoice
'advertiser': name of advertising organization
'contract_num': contract number from invoice
'flight_from': starting flight date
'flight_to': ending flight date

Do not include any other text, no yapping.

"examples": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": "[Image 1: Sample invoice from ABC Advertising, Contract #12345, dates Jan 1-31 2024, total $5,250.00]"
        },
        {
          "type": "text",
          "text": "Extract the invoice data."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\n  \"gross_amount\": 5250.00,\n  \"advertiser\": \"ABC Advertising\",\n  \"contract_num\": \"12345\",\n  \"flight_from\": \"2024-01-01\",\n  \"flight_to\": \"2024-01-31\"\n}"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": "[Image 2: Sample invoice from XYZ Media Group, Contract #INV-2024-789, dates Mar 15 - Apr 14 2024, total $12,800.50]"
        },
        {
          "type": "text",
          "text": "Extract the invoice data."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\n  \"gross_amount\": 12800.50,\n  \"advertiser\": \"XYZ Media Group\",\n  \"contract_num\": \"INV-2024-789\",\n  \"flight_from\": \"2024-03-15\",\n  \"flight_to\": \"2024-04-14\"\n}"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": "[Image 3: Sample invoice from Digital Solutions LLC, Contract #CT-987-B, dates June 1 - June 30 2024, total $8,499.99]"
        },
        {
          "type": "text",
          "text": "Extract the invoice data."
        }
      ]
    },
    {
      "role": "assistant",
      "content": "{\n  \"gross_amount\": 8499.99,\n  \"advertiser\": \"Digital Solutions LLC\",\n  \"contract_num\": \"CT-987-B\",\n  \"flight_from\": \"2024-06-01\",\n  \"flight_to\": \"2024-06-30\"\n}"
    }
]