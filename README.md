# Bangla LLaMA: Bengali Context-Based Question Answering and Retrieval Augment Generation

This repository contains a collection of Llama models fine-tuned for context-based question answering and Bengali retrieval augment generation. 

## Introduction

Bangla LLaMA is derived from LLaMA and trained on various Bengali datasets. These models are designed to provide accurate responses in Bengali with relevant contextual information. They are integrated with the transformers library, making them easy to use for context-based question answering and Bengali retrieval augment generation in your projects.

## Model List

| Model Name | Quantization | Dataset | Hugging Face Link |
|---|---|---|---|
| Bangla LLAMA 1B GGUF 16bit ![New!](https://img.shields.io/badge/New-brightgreen) | 16-bit | [all_combined_bengali_252k](https://huggingface.co/datasets/OdiaGenAI/all_combined_bengali_252k) | [asif00/bangla-llama-1B-gguf-16bit](https://huggingface.co/asif00/bangla-llama-1B-gguf-16bit) |
| Bangla LLAMA 1B Lora ![New!](https://img.shields.io/badge/New-brightgreen) | Lora | [all_combined_bengali_252k](https://huggingface.co/datasets/OdiaGenAI/all_combined_bengali_252k) | [asif00/bangla-llama-1B-lora](https://huggingface.co/asif00/bangla-llama-1B-lora) |
| Bangla LLAMA 1B 4bit ![New!](https://img.shields.io/badge/New-brightgreen) | 4-bit | [all_combined_bengali_252k](https://huggingface.co/datasets/OdiaGenAI/all_combined_bengali_252k) | [asif00/bangla-llama-1B-4bit](https://huggingface.co/asif00/bangla-llama-1B-4bit) |
| Bangla LLAMA | Lora | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama](https://huggingface.co/asif00/bangla-llama) |
| Bangla LLAMA Lora | Lora | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama-lora](https://huggingface.co/asif00/bangla-llama-lora) |
| Bangla LLAMA 4bit | 4-bit | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama-4bit](https://huggingface.co/asif00/bangla-llama-4bit) |
| Bangla LLAMA GGUF 4bit | 4-bit | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama-gguf_q4_k_m](https://huggingface.co/asif00/bangla-llama-gguf_q4_k_m) |
| Bangla LLAMA 16bit | 16-bit | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama-16bit](https://huggingface.co/asif00/bangla-llama-16bit) |
| Bangla LLAMA GGUF 16bit | 16-bit | [alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali) | [asif00/bangla-llama-gguf-16bit](https://huggingface.co/asif00/bangla-llama-gguf-16bit) |

## Datasets Used

* **[all_combined_bengali_252k](https://huggingface.co/datasets/OdiaGenAI/all_combined_bengali_252k):** A large Bengali dataset containing 252,000 samples.
* **[alpaca-cleaned-bengali](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-bengali):** A cleaned version of the Alpaca dataset translated into Bengali, containing 51,760 samples.

## Model Details: LLaMA 3.2 1B Batch

**General Training Details:**

- Language: Bengali
- Use Case: Context-Based Question Answering, Bengali Retrieval Augment Generation

**Specific training details (e.g., training loss, global steps, batch size, epoch) can be found on the respective model's Hugging Face page.**

## Model Details: LLaMA 3 8B Batch

**General Training Details:**

- Language: Bengali
- Use Case: Context-Based Question Answering, Bengali Retrieval Augment Generation

**Specific training details (e.g., training loss, global steps, batch size, epoch) can be found on the respective model's Hugging Face page.**

## How to Use

### GGUF Quantized Models

These models can be loaded and used directly with the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("asif00/bangla-llama-1B-gguf-16bit")
model = AutoModelForCausalLM.from_pretrained("asif00/bangla-llama-1B-gguf-16bit")

# Prepare the input prompt
prompt = "আজ আবহাওয়া কেমন?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text) 
```

### LoRA Models

These models require loading the base model and then applying the LoRA weights using the `peft` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("asif00/bangla-llama-1B")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "asif00/bangla-llama-1B",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load the LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "asif00/bangla-llama-1B-lora"
)

# Prepare the input prompt
prompt = "বাংলাদেশের রাজধানী কোথায়?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text with LoRA (no gradient calculation needed during inference)
with torch.no_grad():
    outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
```

### 4-bit Quantized Models

These models can be loaded and used similarly to GGUF models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("asif00/bangla-llama-1B-4bit")
model = AutoModelForCausalLM.from_pretrained("asif00/bangla-llama-1B-4bit")

# Prepare the input prompt
prompt = "পৃথিবী সূর্যের চারদিকে ঘুরতে কত সময় নেয়?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
```

## General Prompt Structure

```python
prompt = """Below is an instruction in Bengali language that describes a task, paired with an input also in Bengali language that provides further context. Write a response in Bengali language that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""
```

## Example Usage

```python
# ... (load your chosen model and tokenizer as described above) ...

def generate_response(instruction, context):
    # Format the prompt with the instruction and context
    prompt = prompt.format(instruction, context, "") 

    # Tokenize the prompt and move it to the device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate text
    outputs = model.generate(**inputs, max_new_tokens=1024) 

    # Decode the generated output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the response part from the generated text
    response_start = response.find("### Response:") + len("### Response:")
    response = response[response_start:].strip()
    
    # Return the extracted response
    return response

# Example instruction and context
instruction = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর সম্পর্কে একটি সংক্ষিপ্ত বিবরণ দিন।"
context = "মহাশ্বেতা দেবী ২০১৬ সালে হৃদরোগে আক্রান্ত হয়ে কলকাতায় মৃত্যুবরণ করেন।"

# Generate a response using the defined function
answer = generate_response(instruction, context)

# Print the generated answer 
print("উত্তর:", answer)
```

## Fine-tuning script

I have added the original script used for finetuning so you can replicate it. Find the `finetune` script here: [finetune.ipynb](finetune.ipynb)

## Disclaimer

The Bangla LLaMA models have been trained on limited datasets, and their responses may not always be perfect or accurate. The models' performance is dependent on the quality and quantity of the data they have been trained on. Given more resources, such as high-quality data and longer training time, the models' performance can be significantly improved.


