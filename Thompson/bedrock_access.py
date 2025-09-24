import json 
import logging 
import boto3 
from botocore.exceptions import ClientError 
import yaml 
import time 
import pandas as pd 

def generate_meassages(user_data):
    from datetime import datetime
    print(f'##### Function called at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')

    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1" 
    )

    cluster_type = user_data["user_login"][0]["cluster_type"]

    if cluster_type == "HELOC":
        cluster_type = "Home Equity Line of Credit" 
    reasoning = user_data["user_login"][0]["reasoning"]

    print(bedrock_runtime_client)

    model_id = 'amazon.nova-micro-v1:0'
    system_prompt = "You are an expert in encouraging banking customer to engage with either home mortgage purchase, refinancing or home equity loan. I need you to come up with ten brief and powerful messages, each message has at least ten and most thirty words, and be creative for every time you are being called via API, do not use save verbiage each time. Do not make any offers or mention anything numeric, such as years, terms, interest rates, fees."
    
    # Add timestamp to force unique requests
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    user_message = (
        f"This message is for user that is interested in {cluster_type}. "
        f"Reasoning: {reasoning}. "
        f"Request ID: {unique_id} - Generate fresh, unique messages."
    )
    print('##### User Message: ', user_message, '\n')

    json_output_prompt = f"""
{user_message}

Return the output as a JSON array of strings, with each string containing one message. Do not include any extra text.and

```json
"""
    
    inference_configuration = {
        "maxTokens": 1024,
        "temperature": 0.9,
        "topP": 0.9,

    }

    response = bedrock_runtime_client.converse(
        modelId=model_id,
        system = [
            {"text": system_prompt} 
        ],
        messages = [
            {
                "role": "user",
                "content": [{"text": json_output_prompt}] 
            }
        ],
        inferenceConfig = inference_configuration
    )

    print('##### Python Dict: ', response, '\n') 

    json_str = json.dumps(response, indent=2)
    print('##### JSON String: ', json_str, '\n')

    # Extract list of messages from the response
    messages_json_str = response['output']['message']['content'][0]['text'] 
    messages_list = json.loads(messages_json_str) 

    print('##### Messages List: ')
    for i, message in enumerate(messages_list, 1):
        print(f'{i:2d}. {message}')
    print()
    return messages_list