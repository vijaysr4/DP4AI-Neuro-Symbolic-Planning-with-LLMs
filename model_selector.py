import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class BasePlanner:
    """Abstract base planner interface."""
    def generate_plan(self, prompt: str, system_message: str = "") -> str:
        raise NotImplementedError("Subclasses must implement generate_plan.")

class GPT4Planner(BasePlanner):
    """
    Planner using OpenAI's GPT-4.
    Note: GPT-4 is not free; you must have an API key with GPT-4 access.
    """
    def __init__(self, openai_api_key: str, temperature: float = 0.7):
        openai.api_key = openai_api_key
        self.temperature = temperature

    def generate_plan(self, prompt: str, system_message: str = "") -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]

class GPT4OMiniPlanner(BasePlanner):
    """
    Planner using OpenAI's GPT-4O-Mini.
    Replace 'gpt-4o-mini' with the actual model identifier as provided by OpenAI.
    """
    def __init__(self, openai_api_key: str, temperature: float = 0.7):
        openai.api_key = openai_api_key
        self.temperature = temperature

    def generate_plan(self, prompt: str, system_message: str = "") -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]

class GPT4OPlanner(BasePlanner):
    """
    Planner using OpenAI's GPT-4O.
    Replace 'gpt-4o' with the actual model identifier as provided by OpenAI.
    """
    def __init__(self, openai_api_key: str, temperature: float = 0.7):
        openai.api_key = openai_api_key
        self.temperature = temperature

    def generate_plan(self, prompt: str, system_message: str = "") -> str:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message["content"]

class LlamaPlanner(BasePlanner):
    """
    Planner using a LLaMA model loaded from Hugging Face.
    Example model_names: "meta-llama/Meta-Llama-3.1-8B-Instruct", etc.
    """
    def __init__(self, model_name: str, hf_token: str, max_new_tokens: int = 128, temperature: float = 0.7):
        if not hf_token:
            raise ValueError("A valid Hugging Face token is required.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            token=hf_token
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    def generate_plan(self, prompt: str, system_message: str = "") -> str:
        # system_message is ignored for HF pipelines.
        outputs = self.pipe(prompt)
        return outputs[0]["generated_text"]

def get_planner(model_choice: str, **kwargs) -> BasePlanner:
    """
    Returns an instance of a planner based on model_choice.

    Supported model_choice strings (case-insensitive):
      - "gpt4": uses GPT-4 via OpenAI (requires openai_api_key in kwargs)
      - "gpt-4o-mini": uses GPT-4O-Mini via OpenAI (requires openai_api_key in kwargs)
      - "gpt-4o": uses GPT-4O via OpenAI (requires openai_api_key in kwargs)
      - "llama-3.1-8b-instruct": uses the LLaMA 3.1 8B Instruct model
      - "deepseek-r1-7b": uses the DeepSeek-R1-Distill-Qwen-7B model

    Additional keyword arguments:
      - openai_api_key (for OpenAI models)
      - hf_token (for LLaMA models)
      - temperature, max_new_tokens, etc.
    """
    choice = model_choice.strip().lower()
    print(f"DEBUG: Received model_choice = '{model_choice}', normalized to '{choice}'")
    if choice == "gpt4":
        if "openai_api_key" not in kwargs:
            raise ValueError("For GPT-4, you must provide an 'openai_api_key'.")
        return GPT4Planner(
            openai_api_key=kwargs["openai_api_key"],
            temperature=kwargs.get("temperature", 0.7)
        )
    elif choice == "gpt-4o-mini":
        if "openai_api_key" not in kwargs:
            raise ValueError("For GPT-4O-Mini, you must provide an 'openai_api_key'.")
        return GPT4OMiniPlanner(
            openai_api_key=kwargs["openai_api_key"],
            temperature=kwargs.get("temperature", 0.7)
        )
    elif choice == "gpt-4o":
        if "openai_api_key" not in kwargs:
            raise ValueError("For GPT-4O, you must provide an 'openai_api_key'.")
        return GPT4OPlanner(
            openai_api_key=kwargs["openai_api_key"],
            temperature=kwargs.get("temperature", 0.7)
        )
    elif choice == "llama-3.1-8b-instruct":
        if "hf_token" not in kwargs:
            raise ValueError("For LLaMA, you must provide an 'hf_token'.")
        return LlamaPlanner(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            hf_token=kwargs["hf_token"],
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            temperature=kwargs.get("temperature", 0.7)
        )
    elif choice == "deepseek-r1-7b":
        if "hf_token" not in kwargs:
            raise ValueError("For DeepSeek-R1-7B, you must provide an 'hf_token'.")
        return LlamaPlanner(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            hf_token=kwargs["hf_token"],
            max_new_tokens=kwargs.get("max_new_tokens", 128),
            temperature=kwargs.get("temperature", 0.7)
        )
    else:
        print("DEBUG: In the else block. choice was:", choice)
        raise ValueError(f"Unsupported model choice: {model_choice}")
