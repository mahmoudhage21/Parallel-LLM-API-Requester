# Azure OpenAI Parallel Requests Handler

This project simplifies making parallel requests to the Azure OpenAI API for chat completions of scenarios where one needs to batch process a large number of **prepared prompts simultaneously**.



This project efficiently manages rate limits and incorporates robust error handling to streamline processing multiple inputs simultaneously. Unlike the official OpenAI parallel implementation, which can be complex and cumbersome for beginners, this project offers a simplified, easy-to-understand approach.

## Example

For a very simple scenario where the data consists of 100 requests asking simple questions such as `What is 1+1?`, `What is 5+5?`, processing these requests one by one took about 18.6 seconds 🛵. However, using the parallel processing method, this time was significantly reduced to approximately 2.6 seconds 🏎️, making it 7 times faster.


So hit it with more complex requests and larger datasets, and watch this method flexes its muscles, shaving off loads of time and zipping through tasks like a rocket booster 🚀

## Requirements

- API key from Azure OpenAI
- Store the API key in a file named .env `AZURE_OPENAI_API_KEY = <your_token>`

## Installation

Set up a virtual environment (macOS) as a kernel in Jupyter Notebook by installing the required packages to get started with this project:

```bash
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

python -m ipykernel install --user --name=myenv --display-name="Python 3.11 (myenv)"
```

## Usage

To use this implementation, structure your input data as follows and utilize the provided APIPlayer class to handle parallel requests:

### Data Format Example

```bash
[
 [{'role': 'system', 'content': "<Replace this with your desired system msg>"},
  {'role': 'user', 'content': '<Replace this with your desired user msg>'}],

 [{'role': 'system', 'content': "<Replace this with your desired system msg>"},
  {'role': 'user', 'content': '<Replace this with your desired user msg>'}],

 ...
]
```

### Sample Class Usage

Instantiate the APIRequestor class and call the get_responses_parallel method with your input data:

```bash
gpt35_turbo_api = APIRequester(model_name = "gpt-35-turbo", temperature = 1.0, max_tokens = 20, rate_limit = 100) 
results = gpt35_turbo_api.get_responses_parallel(message_sequences)
results[:2]
```

Each result is saved as a dictionary with input (the user's request message) and content (the response from the API), maintaining the relationship between each request and its corresponding response.

```bash
[{'input': 'What is 53 + 53?', 'content': '106'},
 {'input': 'What is 100 + 100?', 'content': '200'}]
```

### Key Features

- [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html): Manages multiple requests in parallel, improving response time.
- [Semaphore](https://docs.python.org/3/library/threading.html#semaphore-example): Controls the rate of API calls to comply with rate limits.
- [Retry Mechanism](https://tenacity.readthedocs.io/en/latest/): Handles intermittent errors effectively by automatically retrying failed requests.
- [Custom error handling](https://tenacity.readthedocs.io/en/latest/index.html?highlight=retry_error_callback#custom-callbacks): Provides a fallback mechanism that triggers after all retry attempts fail, allowing the process to proceed smoothly despite errors.

## Related Projects

While other projects provide mechanisms to interact with OpenAI's API, this project focuses on simplicity and ease of use, especially for users new to parallel computing:

This Script [openai-cookbook/examples/api_request_parallel_processor.py](https://github.com/openai/openai-cookbook/blob/970d8261fbf6206718fe205e88e37f4745f9cf76/examples/api_request_parallel_processor.py) is well-suited for making parallel requests to the OpenAI API. However, it can be complex and cumbersome for scenarios where one wants to just send a lot of prompts that are already prepared simultaneously. This project aims to streamline and simplify that process.



## Credits

Special thanks to the Max Planck Institute for Human Development, Center for Humans & Machines for providing the Azure OpenAI API endpoint that facilitated the development of this project.

For more information on their work and further research, please visit their [GitHub](https://github.com/center-for-humans-and-machines) and [official website](https://www.mpib-berlin.mpg.de/chm).

