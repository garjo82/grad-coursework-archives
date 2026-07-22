import json
from pathlib import Path

from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    """
    Generate the offline RFT dataset using the CoT model.
    Processes prompts individually to prevent VRAM exhaustion on the 1080 Ti.
    """
    model = CoTModel()
    train_data = Dataset("train")

    rft_data = []

    # Process individually to strictly control the memory footprint
    for i in tqdm(range(len(train_data)), desc="Generating RFT Data"):
        question, correct_float = train_data[i]

        # CoTModel formats this into the chat template for optimal reasoning
        prompt = model.format_prompt(question)

        # batched_generate with sequences > 1 returns a list of lists.
        # Since we are passing 1 prompt, we index [0] to get the list of completions.
        completions = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature
        )[0]

        # Search for the first mathematically correct reasoning chain
        for completion in completions:
            parsed_ans = model.parse_answer(completion)
            if is_answer_valid(parsed_ans, correct_float):
                # Save the successful prompt + reasoning + answer format
                rft_data.append([question, correct_float, completion])
                break

    # Save to the specific directory expected by the Dataset class
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(rft_data, f, indent=2)

    print(f"\nGenerated {len(rft_data)} valid CoT examples out of {len(train_data)} prompts.")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)