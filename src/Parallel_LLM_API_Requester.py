import os
import time
import json
import tiktoken
import threading
from pydantic import BaseModel
from threading import Semaphore
from typing import List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)


class ParallelAPIRequesterConfig(BaseModel):
    provider: Literal["openai", "azure"]
    model_name: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 1.0
    request_rate_limit: Optional[int] = 50
    token_rate_limit: Optional[int] = 7500
    cost_print_interval: Optional[int] = 50
    check_Intent_keys_and_values: Optional[bool] = False
    budget: Optional[float] = 30 # in dollars
    return_in_json_format: Optional[bool] = True


class TokenSemaphore:
    '''
    
    Custom TokenSemaphore for Token Rate Limiting

    The Python standard Semaphore from the threading module starts with an internal counter, which you specify upon creation. This counter decrements each time acquire() is called and increments when release() is called.
    However, the standard Semaphore doesn't support acquiring or releasing more than one unit of the counter at a time, which means it can't directly manage multiple tokens per request out-of-the-box if those requests consume a variable number of tokens.

    The following custom class allows you to specify how many tokens to acquire or release at a time, giving you the flexibility needed for handling variable token counts per API request.
    '''

    def __init__(self, max_tokens):
        self.tokens = max_tokens
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def acquire(self, required_tokens):
        with self.lock:
            while self.tokens < required_tokens:
                self.condition.wait()
            self.tokens -= required_tokens

    def release(self, released_tokens):
        with self.lock:
            self.tokens += released_tokens
            self.condition.notify_all()


class ParallelAPIRequester:
    def __init__(
        self,
        name,
        config: ParallelAPIRequesterConfig,
    ):
        """
        This class is designed to handle parallel API requests via specified providers: Azure or OpenAI.

        When using Azure:
        - You must adjust the 'model_name' to match the models deployed in your specific Azure environment.
        - Pricing may also vary depending on your Azure subscription and should be set accordingly in the 'price_list' based on your contract.

        When using OpenAI:
        - The 'model_calls' used are standard across the API, but the allowed request rates and token rates depend on your subscription tier.
        - It's essential to check your tier limitations and adjust the 'request_rate_limit' and 'token_rate_epoch' in the config to comply with the limits of your specific OpenAI API tier.

        Ensure the appropriate values are configured to avoid API request or token request rejections due to limit overreaches.
        """
        assert config.request_rate_limit is not None, "request_rate_limit must be provided in the config"
        assert config.token_rate_limit is not None, "token_rate_limit must be provided in the config"

        self.provider = config.provider
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

        self.request_rate_limit = config.request_rate_limit  # Requests rate per minute
        self.request_semaphore = Semaphore(self.request_rate_limit)
        self.token_rate_limit = config.token_rate_limit  # Token rate limit per minute
        self.token_semaphore = TokenSemaphore(self.token_rate_limit)  # Custom semaphore for token limit

        self.cost_print_interval = config.cost_print_interval # print cost every k completed requests
        self.return_in_json_format = config.return_in_json_format

        self.budget = config.budget
        self.input_token_count = 0
        self.output_token_count = 0
        self.cost = 0

        
        if self.provider == "azure":
            from openai import AzureOpenAI

            # gets the API Key from environment variable AZURE_OPENAI_API_KEY
            self.client = AzureOpenAI(
                api_version="2024-02-15-preview",
                azure_endpoint= "https://chm-openai.openai.azure.com/",
            )   
            self.price_list = {
                "gpt-35-turbo": {
                    "input": 0.0005 * 0.001,
                    "output": 0.0015 * 0.001,
                },
                "gpt-4-turbo-1106": {
                    "input": 0.01 * 0.001,
                    "output": 0.03 * 0.001,
                }
            }

            available_models = ["gpt-35-turbo", "gpt-4-turbo-1106"]
            assert self.model_name in available_models, f"Model {self.model_name} is not supported. Please use one of {available_models} as the model name."

            if self.model_name == "gpt-35-turbo":
                max_request_rate_limit = 300
                max_token_rate_limit = 50e3
                assert self.request_rate_limit <= max_request_rate_limit, f"Request rate limit for {self.model_name} in our Azure subscription is 300 requests per minute. Please set the request rate limit to a value less than {max_request_rate_limit}."
                assert self.token_rate_limit <= max_token_rate_limit, f"Token rate limit for {self.model_name} in our Azure subscription is 50000 tokens per minute. Please set the token rate limit to a value less than {max_token_rate_limit}."
            
            elif self.model_name == "gpt-4-turbo-1106":
                max_request_rate_limit = 60
                max_token_rate_limit = 10e3
                assert self.request_rate_limit <= max_request_rate_limit, f"Request rate limit for {self.model_name} in our Azure subscription is 30 requests per minute. Please set the request rate limit to a value less than {max_request_rate_limit}."
                assert self.token_rate_limit <= max_token_rate_limit, f"Token rate limit for {self.model_name} in our Azure subscription is 10000 tokens per minute. Please set the token rate limit to a value less than {max_token_rate_limit}."
        
        elif self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            self.price_list = {
                "gpt-3.5-turbo": {
                    "input": 0.0005 * 0.001,
                    "output": 0.0015 * 0.001,
                },
                "gpt-4-turbo": {
                    "input": 0.01 * 0.001,
                    "output": 0.03 * 0.001,
                }
            }

            available_models = [model.id for model in self.client.models.list().data]
            assert self.model_name in available_models, f"Model {self.model_name} is not available in the OpenAI API. Please use one of the following models: {', '.join(available_models)} as the model name."
            
            if self.model_name == "gpt-3.5-turbo":
                max_request_rate_limit = 5e3
                max_token_rate_limit = 160e3
                assert self.request_rate_limit <= max_request_rate_limit, f"Request rate limit for {self.model_name} in our Azure subscription is 300 requests per minute. Please set the request rate limit to a value less than {max_request_rate_limit}."
                assert self.token_rate_limit <= max_token_rate_limit, f"Token rate limit for {self.model_name} in our Azure subscription is 50000 tokens per minute. Please set the token rate limit to a value less than {max_token_rate_limit}."
            
            elif self.model_name == "gpt-4-turbo": 
                max_request_rate_limit = 5e3
                max_token_rate_limit = 600e3
                assert self.request_rate_limit <= max_request_rate_limit, f"Request rate limit for {self.model_name} in our Azure subscription is 30 requests per minute. Please set the request rate limit to a value less than {max_request_rate_limit}."
                assert self.token_rate_limit <= max_token_rate_limit, f"Token rate limit for {self.model_name} in our Azure subscription is 10000 tokens per minute. Please set the token rate limit to a value less than {max_token_rate_limit}."
        else:
            raise ValueError(f"Provider {self.provider} is not supported. Please use 'azure' or 'openai' as the provider.")

        assert self.model_name in self.price_list.keys(), f"please add the price for the model {self.model_name} in the price_list in the ParallelAPIRequester class."
    
    # Adapted from https://github.com/openai/openai-cookbook/blob/970d8261fbf6206718fe205e88e37f4745f9cf76/examples/api_request_parallel_processor.py#L339-L389
    def num_tokens_consumed_from_request(
            self,
            request_json: List,
    ):
        # for gpt-4 / gpt-3.5-turbo, the encoding is "cl100k_base"
        encoding = tiktoken.get_encoding("cl100k_base")
        n = 1 # number of completions 
        completion_tokens = n * 500 # self.max_tokens 
        # chat completions
        num_tokens = 0
        try:
            for message in request_json:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        except KeyError:
            print(f"Invalid request JSON: {request_json}")
            return 0


    def handle_last_retry_error(retry_state):
        print(f"All retry attempts failed for: {retry_state.args[0]}\nReturning None for this request.")
        return None  # Custom behavior after all retries fail


    @retry(wait=wait_fixed(2) + wait_random(10, 20),
            stop=stop_after_attempt(2),
            before_sleep= lambda retry_state: print("Retrying..."),
            retry_error_callback=handle_last_retry_error)
    def get_response(self, system_user_message: List):
        estimated_tokens = self.num_tokens_consumed_from_request(system_user_message)
        # Acquire both semaphores to manage requests and tokens
        self.request_semaphore.acquire()
        try:
            self.token_semaphore.acquire(estimated_tokens)  # Acquire tokens
            try:
                response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=system_user_message,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            response_format={"type": "json_object"} if self.return_in_json_format else None,
                        )
                generated_response = response.choices[0].message.content
                if self.return_in_json_format:
                    generated_response = json.loads(generated_response) # will raise an error if not valid JSON and trigger a retry
                
                self.input_token_count += response.usage.prompt_tokens
                self.output_token_count += response.usage.completion_tokens
                self.cost += response.usage.prompt_tokens * self.price_list[self.model_name]["input"] + response.usage.completion_tokens * self.price_list[self.model_name]["output"]

                return generated_response
            except json.JSONDecodeError:
                print(f"Invalid JSON response for message {system_user_message}")
                raise  # This re-raises the last exception, triggering a retry
            except Exception as e:
                print(f"Error while processing: {str(e)}")
                raise
            finally:
                self.token_semaphore.release(estimated_tokens)  # Release tokens after the request
        finally:
            self.request_semaphore.release()  # Release request semaphore after handling tokens
            time.sleep(60 / self.request_rate_limit)  # Pause to respect the rate limit to evenly distribute requests over time

    
    def get_responses_parallel(self, messages_list):
        results = []
        started = time.time()
        request_counter = 0
        # assert that each item in the list has the key api_message of the form [{"role": "system", "content": "system message"}, {"role": "user", "content": "user message"}]
        assert all("api_message" in item for item in messages_list), "Each item in the list must have the key api_message"
        identifier_keys = [key for key in messages_list[0] if key != "api_message"]

        with ThreadPoolExecutor(max_workers=self.request_rate_limit) as executor:
            # Submit tasks to the executor, associating each future with its corresponding index and message
            future_to_info = {executor.submit(self.get_response, item['api_message']): item for item in messages_list}
            for future in as_completed(future_to_info):
                item = future_to_info[future]

                # Increment the request counter after each successful request
                request_counter += 1
                try:
                    result = future.result()
                    # Extract the user's query from the message (assuming the response format is JSON string like '{"response": "4"}')
                    user_query = next((msg['content'] for msg in item['api_message'] if msg['role'] == 'user'), "Unknown query")
                    system_query = next((msg['content'] for msg in item['api_message'] if msg['role'] == 'system'), "Unknown query")

                    result_dict = {key: item[key] for key in identifier_keys}
                    result_dict.update({"api__system_message": system_query, "api__user_message": user_query, "response": result})
                    results.append(result_dict)


                except Exception as e:
                    result_dict = {key: item[key] for key in identifier_keys}
                    result_dict.update({"api__system_message": "Error", "api__user_message": "Error", "response": f"Error processing message: {str(e)}"})
                    results.append(result_dict)
                
                # Print the total cost every k requests
                if request_counter % self.cost_print_interval == 0 or request_counter == len(messages_list):
                    print(f"Total cost after {request_counter}/{len(messages_list)} requests: {self.cost} dollars")
                
                # Check if the budget has exceeded the maximum allowed cost
                if self.cost >= self.budget:
                    print(f"Budget of {self.budget} dollars exceeded. Stopping further requests.")
                    break

        elapsed_time = time.time() - started
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    
        return results