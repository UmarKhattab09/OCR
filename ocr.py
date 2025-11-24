import ollama
model_path = "E:/ollamamodels"
# Simple text generation
def generate_text_with_gemma(prompt):
    """Generates text using Gemma 3 12B with a given prompt."""
    try:
        response = ollama.generate(
            model='gemma3:12b',
            prompt=prompt
        )
        return response['response']
    except Exception as e:
        return f"Error generating text: {e}"

# Conversational interface
def chat_with_gemma(messages):
    """Engages in a conversational chat with Gemma 3 12B."""
    try:
        chat_response = ollama.chat(
            model='gemma3:12b',
            messages=messages
        )
        return chat_response['message']['content']
    except Exception as e:
        return f"Error in chat: {e}"

if __name__ == "__main__":
    # Example 1: Simple text generation
    prompt_for_generation = "Explain the concept of quantum entanglement in simple terms."
    generated_text = generate_text_with_gemma(prompt_for_generation)
    print("--- Generated Text ---")
    print(generated_text)
    print("\n")

    # Example 2: Conversational interface
    chat_history = [
        {'role': 'user', 'content': 'What is the capital of France?'},
        {'role': 'assistant', 'content': 'The capital of France is Paris.'},
        {'role': 'user', 'content': 'Can you tell me more about its history?'}
    ]
    chat_response_content = chat_with_gemma(chat_history)
    print("--- Conversational Chat ---")
    print(chat_response_content)