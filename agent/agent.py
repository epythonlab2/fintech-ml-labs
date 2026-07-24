from tools import get_current_time, calculator
import ollama


class SimpleAgent:
    def think(self, prompt: str) -> str:
        normalized_prompt = prompt.lower()

        # Tool: Current Time
        if "time" in normalized_prompt:
            return get_current_time()

        # Tool: Calculator
        if "calculate" in normalized_prompt:
            expression = normalized_prompt.replace("calculate", "").strip()
            return calculator(expression)

        # Fall back to the LLM
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]