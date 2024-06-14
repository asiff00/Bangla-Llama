
# This is a collection of Llama 8B models I fine-tuned for context based questions answering. 

Bangla LLaMA is a specialized model for context-based question answering and Bengali retrieval augment generation. It is derived from LLaMA 3 8B and trained on the iamshnoo/alpaca-cleaned-bengali dataset. This model is designed to provide accurate responses in Bengali with relevant contextual information. It is integrated with the transformers library, making it easy to use for context-based question answering and Bengali retrieval augment generation in projects. 

# Model list:

### Lora Quantization:
- [Bangla LLAMA](https://huggingface.co/asif00/bangla-llama)
- [Bangla LLAMA Lora](https://huggingface.co/asif00/bangla-llama-lora)

### 4-bit Quantization:
- [Bangla LLAMA 4bit](https://huggingface.co/asif00/bangla-llama-4bit)
- [Bangla LLAMA GGUF 4bit](https://huggingface.co/asif00/bangla-llama-gguf_q4_k_m)

### 16-bit Quantization:
- [Bangla LLAMA 16bit](https://huggingface.co/asif00/bangla-llama-16bit)
- [Bangla LLAMA GGUF 16bit](https://huggingface.co/asif00/bangla-llama-gguf-16bit)


# Model Details:

- Model Family: Llama 3 8B
- Language: Bengali
- Use Case: Context-Based Question Answering, Bengali Retrieval Augment Generation
- Dataset: iamshnoo/alpaca-cleaned-bengali (51,760 samples)
- Training Loss: 0.4038
- Global Steps: 647
- Batch Size: 80
- Epoch: 1


# How to Use:

You can use the model with a pipeline for a high-level helper or load the model directly. Here's how:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("question-answering", model="asif00/bangla-llama")
```

```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("asif00/bangla-llama")
model = AutoModelForCausalLM.from_pretrained("asif00/bangla-llama")
```

# General Prompt Structure: 

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

# To get a cleaned up version of the response, you can use the `generate_response` function:

```python
def generate_response(question, context):
    inputs = tokenizer([prompt.format(question, context, "")], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response_start = responses.find("### Response:") + len("### Response:")
    response = responses[response_start:].strip()
    return response
```

# Example Usage:

```python
question = "ভারতীয় বাঙালি কথাসাহিত্যিক মহাশ্বেতা দেবীর মৃত্যু কবে হয় ?"
context = "২০১৬ সালের ২৩ জুলাই হৃদরোগে আক্রান্ত হয়ে মহাশ্বেতা দেবী কলকাতার বেল ভিউ ক্লিনিকে ভর্তি হন। সেই বছরই ২৮ জুলাই একাধিক অঙ্গ বিকল হয়ে তাঁর মৃত্যু ঘটে। তিনি মধুমেহ, সেপ্টিসেমিয়া ও মূত্র সংক্রমণ রোগেও ভুগছিলেন।"
answer = generate_response(question, context)
print(answer)
```

# Fine-tuning script:
I have added the original script used for finetuning so you can replicate it. Find the `finetune` script here: [finetune.ipynb](finetune.ipynb)

# Disclaimer:

The Bangla LLaMA-4bit model has been trained on a limited dataset, and its responses may not always be perfect or accurate. The model's performance is dependent on the quality and quantity of the data it has been trained on. Given more resources, such as high-quality data and longer training time, the model's performance can be significantly improved.


# Resources: 
Work in progress...
