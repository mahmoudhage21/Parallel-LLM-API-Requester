

The purpose of this code is to make parallel requests to the Azure OpenAI API for chat completions, effectively managing rate limits and ensuring robust error handling. It's designed for scenarios where multiple inputs need to be processed simultaneously, reducing the overall time required to obtain responses from the API.


## Set Up a Virtual Environment as a kernel in Jupyter Notebook (macOS)

Python version: 3.11.4
```bash
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

python -m ipykernel install --user --name=myenv --display-name="Python 3.11 (myenv)"
```


## Implementation Overview

### Data Format Example

The input data, should be structured as a list of message dictionaries, where each message has a `role` (either "system" or "user") and `content`. Here’s a very simple example:

```bash
[
 [{'role': 'system', 'content': "You are a helpful AI assistant.  Always answer in the following json format: {'content': '2'}."},
  {'role': 'user', 'content': 'What is 1 + 1?'}],

 [{'role': 'system', 'content': "You are a helpful AI assistant.  Always answer in the following json format: {'content': '2'}."},
  {'role': 'user', 'content': 'What is 2 + 2?'}],

  ...
]

```
### ThreadPoolExecutor for Parallel Requests:

Utilizes `ThreadPoolExecutor` to send multiple requests in parallel, significantly speeding up the process when dealing with a large number of requests.

With `max_workers` you can set a rate limit to prevent overwhelming the API with too many simultaneous requests.

[More info here](https://docs.python.org/3/library/concurrent.futures.html)

### Semaphore for Rate Limiting:

A semaphore is used to enforce a maximum number of concurrent requests, as per the API's rate limit, preventing rate limit violations.

[More info here](https://docs.python.org/3/library/threading.html#semaphore-example)

### time.sleep for Request Spacing:

After each request, the thread pauses (using `time.sleep(60 / self.rate_limit)`) to evenly distribute requests over time, aligning with the permitted rate limit.

### Retry Mechanism:

Implements a retry mechanism for requests that fail due to recoverable errors, using a combination of fixed and random wait times between retries to avoid immediate repeat failures.

[More info here](https://tenacity.readthedocs.io/en/latest/)
### Error Handling:

Custom error handling logic (`handle_retry_error`) provides a fallback action after all retry attempts fail (`return None`), ensuring the process can continue.

A try-except block catches and handles `json.JSONDecodeError` to manage responses that do not return valid JSON, attempting retries as necessary.

### Result Formatting and Association:

Each result is saved as a dictionary with `input` (the user's request message) and `content` (the response from the API), maintaining a clear association between the input and its corresponding output to that even though requests are processed in parallel, the relationship between requests and responses is preserved.