from transformers import pipeline

def get_transformer_model(model_id):
    return pipeline('text-generation', 
                    model=model_id,
                    device_map="auto",
                    load_in_4bit_mode=True)

def generate_text(model, prompt, max_length=250):
    return model(prompt, max_length=max_length)[0]['generated_text']

model = get_transformer_model(
    model_id="meta-llama/Llama-3.2-3B-Instruct")

prompt = "How to make a cake"
print(generate_text(model, prompt))
