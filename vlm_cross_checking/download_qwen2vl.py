import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
huggingface_model_dir = os.path.join(cur_dir, "huggingface_model")
hf_home = os.path.join(huggingface_model_dir, "hf_home")
model_dir = os.path.join(huggingface_model_dir, "model")
processor_dir = os.path.join(huggingface_model_dir, "processor")

os.makedirs(huggingface_model_dir, exist_ok=True)

os.makedirs(hf_home, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(processor_dir, exist_ok=True)

os.environ['HF_HOME'] = hf_home

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2-VL-7B-Instruct"

print(f"Downloading {model_name}...")

print("Downloading processor...")
processor = AutoProcessor.from_pretrained(model_name)
processor.save_pretrained(processor_dir)
print(f"Processor saved to {processor_dir}")

print("Downloading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    trust_remote_code=True
)
model.save_pretrained(model_dir)
print(f"Model saved to {model_dir}")

print("Download complete!")