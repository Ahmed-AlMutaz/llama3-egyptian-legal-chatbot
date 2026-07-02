# llama3-egyptian-legal-chatbot

# ⚖️ Llama-3.1-8B Egyptian Legal Chatbot (Fine-Tuning & MLOps)

An end-to-end Large Language Model (LLM) fine-tuning pipeline that adapts **Meta-Llama-3.1-8B-Instruct** to understand complex Egyptian legal structures and respond naturally in the **Egyptian Arabic dialect**. 

This project utilizes memory-efficient optimization techniques via **Unsloth** and **LoRA**, with real-time training tracking managed through **Weights & Biases (W&B)**.

---

## 🚀 Key Features
* **Dialect Adaptation:** Fine-tuned specifically on Egyptian legal documents, contracts, and localized text pairs.
* **Memory & Speed Optimization:** Accelerated training using **Unsloth**, implementing 4-bit quantization and LoRA adapters to drastically reduce VRAM usage.
* **MLOps Integration:** Fully tracked training runs utilizing **Weights & Biases** for monitoring training loss, learning rate decay, and GPU compute metrics.
* **Production-Ready Deployment:** Uploaded a standalone, 16-bit fully merged model to Hugging Face for seamless inference.

---

## 📊 Training & MLOps (Weights & Biases)
The training process was rigorously monitored using W&B to ensure convergence and prevent gradient explosions. 

* **Base Model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
* **LoRA Configuration:** `r=16`, `lora_alpha=16`, `target_modules` (q, k, v, o, gate, up, down)
* **Hyperparameters:** `Learning Rate: 2e-4`, `Optimizer: adamw_8bit`, `Batch Size: 2`, `Gradient Accumulation Steps: 8`

> 💡 *Tip: Replace the placeholders below with your actual W&B chart and chat UI screenshots saved in your images folder!*

### Training Loss Convergence:
![Training Loss](images/loss_chart.png)

### GPU Hardware Metrics:
![GPU Performance](images/system_chart.png)

---

## 💬 Interactive Demo
Here is a live preview of the model generating precise legal responses in pure Egyptian dialect without any template artifacts:

![Chatbot Demo](images/chat_demo.png)

---

## 🛠️ Quick Start / Inference

You can pull the fully merged model directly from Hugging Face and run inference using standard `transformers` code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Ahmed792/lora_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "إيه المقصود بالمصلحة العامة لما بنتكلم عن الملكية الفكرية؟"}
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.5)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
