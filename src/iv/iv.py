from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

def get_transformer_model(model_id):
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
        return pipe
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please ensure the model ID is correct and the model is downloaded.")
        return None

def generate_text(model, prompt, max_length=250):
    if model is None:
        return "Model not available."
    return model(prompt, max_length=max_length)[0]['generated_text']

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = get_transformer_model(model_id)

if model:
    prompt = "Once upon a time"
    generated_text = generate_text(model, prompt)
    print(generated_text)
else:
    print("Failed to load the model.")

