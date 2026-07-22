from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Define structured dialogue for in-context chain-of-thought learning
        messages = [
            {
                "role": "system",
                "content": "You are an expert unit conversion assistant. Be concise. Provide a brief step-by-step reasoning chain and wrap your final numerical answer explicitly inside <answer> and </answer> tags."
            },
            {
                "role": "user",
                "content": "How many centimeters are there in 5.5 meters?"
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeters. 5.5 * 100 = 550. <answer>550.0</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        # Apply the SmolLM2 specific chat template tokens without converting to token IDs yet
        formatted_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_string


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})