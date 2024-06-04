# OpenAI Parallel Requests Handler


This project simplifies making parallel requests to the either OpenAI or Azure API for chat completions of scenarios where one needs to batch process a large number of **prepared prompts simultaneously**.



This project efficiently manages rate limits (Requests RPM & Tokens TRM) and incorporates robust error handling to streamline processing multiple inputs simultaneously. Unlike the official OpenAI parallel implementation, which can be complex and cumbersome for beginners, this project offers a simplified, easy-to-understand approach, using libraries such as tenacity and threading.

### Notebook Examples

1. Sentiment Analysis (1000 Requests in less than 2 minutes 🚀🚀🚀)

[This notebook](/sentiment_analysis_API.ipynb) performs sentiment analysis on financial auditor report sentences using the `GPT-3.5 Turbo` model from Azure's OpenAI service. The goal is to categorize the sentiment of each sentence as positive, neutral, or negative. A subset of `1000` samples from the "auditor_sentiment" dataset, available on Hugging Face's datasets hub, is utilized for this analysis. Make sure to adjust the API parameters in the corresponding [config file](src/configs/sentiment_analysis.yml)



## Requirements

- API key from either OpenAI or Azure
- Store the API key in a file named .env `OPENAI_API_KEY= <your_token>` or `AZURE_OPENAI_API_KEY = <your_token>`

## Installation

Set up a virtual environment (macOS) as a kernel in Jupyter Notebook by installing the required packages to get started with this project:

```bash
python -m venv myenv

source myenv/bin/activate

pip install -r requirements.txt

python -m ipykernel install --user --name=myenv --display-name="Python 3.11 (myenv)"
```


### Key Features

- [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html): Manages multiple requests in parallel, improving response time.
- [Semaphore](https://docs.python.org/3/library/threading.html#semaphore-example): Controls the rate of API calls to comply with rate limits.
- [Retry Mechanism](https://tenacity.readthedocs.io/en/latest/): Handles intermittent errors effectively by automatically retrying failed requests.
- [Custom error handling](https://tenacity.readthedocs.io/en/latest/index.html?highlight=retry_error_callback#custom-callbacks): Provides a fallback mechanism that triggers after all retry attempts fail, allowing the process to proceed smoothly despite errors.

## Related Projects

While other projects provide mechanisms to interact with OpenAI's API, this project utilises libraries such as tenacity and threading, focusing on simplicity and ease of use, especially for users new to parallel computing.

This Script [openai-cookbook/examples/api_request_parallel_processor.py](https://github.com/openai/openai-cookbook/blob/970d8261fbf6206718fe205e88e37f4745f9cf76/examples/api_request_parallel_processor.py) is well-suited for making parallel requests to the OpenAI API. However, it can be complex and cumbersome for scenarios where one wants to just send a lot of prompts that are already prepared simultaneously. This project aims to streamline and simplify that process.



## Credits

Special thanks to the Max Planck Institute for Human Development, Center for Humans & Machines for providing the OpenAI and Azure API endpoint that facilitated the development of this project.

For more information on their work and further research, please visit their [GitHub](https://github.com/center-for-humans-and-machines) and [official website](https://www.mpib-berlin.mpg.de/chm).

