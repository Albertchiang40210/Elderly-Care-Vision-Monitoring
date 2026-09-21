import ollama
import time

start_time = time.time()
try:
    response = ollama.chat(
        model="qwen2.5vl:latest",
        messages=[{'role': 'user', 'content': 'Hello, tell me a short story.'}]
    )
    print("Response:", response['message']['content'])
except Exception as e:
    print("Error:", e)
print("Time taken:", time.time() - start_time)
