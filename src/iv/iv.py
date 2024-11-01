from transformers import pipeline
import numpy as np
import stt
import threading

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
        return 

def generate_text(model, prompt, max_length=250):
    if model is None:
        print("Model not available.")
        return 
    return model(prompt, max_length=max_length)[0]['generated_text']

model_id = "meta-llama/Llama-3.2-3B-Instruct"
model = get_transformer_model(model_id)

print("Hi, I'm IV. I can generate text based on your prompt.")
print("ctrl+c to stop recording.")

# Record audio
# Start recording in a separate thread
recording_thread = threading.Thread(target=stt.record_audio())
recording_thread.start()

audio = np.concatenate(stt.recorded_audio, axis=0)  # Combine chunks

if model:
    prompt = "Once upon a time"
    generated_text = generate_text(model, prompt)
    print(generated_text)
else:
    print("Failed to load the model.")

