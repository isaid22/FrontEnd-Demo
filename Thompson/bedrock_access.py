import json 
import logging 
import boto3 
from botocore.exceptions import ClientError 
import yaml 
import time 
import pandas as pd 
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import random as _random
import time as _time

def generate_meassages(user_data, user_message):
    
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

def get_embeddings_batch(
	texts: list,
	model_id: str = "amazon.titan-embed-text-v1",
	region: str = "us-east-1",
	dimensions: int = 1536,
	max_workers: int = 4,
	retries: int = 2,
	base_backoff: float = 0.5,
):
	"""Batch embed multiple texts with Titan, preserving order.

	Args:
		texts: list of strings to embed.
		model_id: Titan embedding model id.
		region: AWS region for Bedrock.
		dimensions: Output dimension size. Supported values depend on model:
			- amazon.titan-embed-text-v1: 1536 (fixed)
			- amazon.titan-embed-text-v2:0: 256, 512, 1024 (default: 1024)
		max_workers: parallel requests.
		retries: retry attempts per item on throttling/transient errors.
		base_backoff: seconds for exponential backoff base.

	Returns:
		List of embedding vectors (list[float]) aligned with input order.
	"""
	if not isinstance(texts, list):
		raise TypeError("texts must be a list of strings")

	client = boto3.client("bedrock-runtime", region_name=region)
	results: list = [None] * len(texts)

	def _embed_one(idx: int, text: str):
		if not isinstance(text, str):
			raise TypeError(f"texts[{idx}] is not a string")
		attempt = 0
		while True:
			try:
				# Build request body based on model capabilities
				if "v2" in model_id:
					# Titan v2 supports configurable dimensions
					body = json.dumps({
						"inputText": text,
						"dimensions": dimensions
					})
				else:
					# Titan v1 has fixed 1536 dimensions
					if dimensions != 1536 and idx == 0:  # Only warn once
						print(f"Warning: Titan v1 only supports 1536 dimensions, ignoring dimensions={dimensions}")
					body = json.dumps({"inputText": text})
				
				resp = client.invoke_model(
					modelId=model_id,
					body=body,
					accept="application/json",
					contentType="application/json",
				)
				payload = json.loads(resp["body"].read())
				return idx, payload.get("embedding", [])
			except ClientError as e:
				code = e.response.get("Error", {}).get("Code")
				status = (e.response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")
				if code in ("ThrottlingException", "TooManyRequestsException") or status in (429, 503, 500):
					if attempt >= retries:
						raise
					delay = base_backoff * (2 ** attempt) + _random.uniform(0, 0.25)
					_time.sleep(delay)
					attempt += 1
					continue
				raise
			except Exception:
				if attempt >= retries:
					raise
				delay = base_backoff * (2 ** attempt) + _random.uniform(0, 0.25)
				_time.sleep(delay)
				attempt += 1

	with ThreadPoolExecutor(max_workers=max_workers) as ex:
		futures = [ex.submit(_embed_one, i, t) for i, t in enumerate(texts)]
		for fut in as_completed(futures):
			idx, vec = fut.result()
			results[idx] = vec

	return results