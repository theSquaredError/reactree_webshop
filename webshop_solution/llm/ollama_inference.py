import requests

def ollama_infer(prompt, model="my-llama3-gguf"):
    """
    Sends a prompt to a local Ollama server and returns the response.
    Args:
        prompt (str): The prompt to send to Ollama.
        model (str): The model to use (default: llama2).
    Returns:
        str: The response from Ollama.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        return f"Error: Ollama request failed: {e}"
