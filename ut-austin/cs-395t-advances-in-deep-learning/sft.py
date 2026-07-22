from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Rounding to 4 decimal places keeps precision high enough for autograders
    # but removes unnecessary noise for the LLM to learn.
    return {
        "question": prompt,
        "answer": f"<answer>{round(float(answer), 4)}</answer>"
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "homework/sft_model",
    **kwargs,
):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    # 1. Load the training data and base model
    train_data = Dataset("train")
    llm = BaseLLM()

    # 2. Format the dataset
    train_dataset = TokenizedDataset(llm.tokenizer, train_data, format_example)

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=10,
        lora_alpha=40,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA and enable gradients for checkpointing
    model = get_peft_model(llm.model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 4. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=16,  # Halved for VRAM safety
        gradient_accumulation_steps=2,   # Re-establishes an effective batch size of 32
        learning_rate=2e-4,
        num_train_epochs=8,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        optim="adamw_torch"
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 6. Train and save the LoRA adapter
    trainer.train()
    trainer.save_model(output_dir)

    # Run evaluation on the newly trained model
    test_model(output_dir)


def test_model(ckpt_path: str = "homework/sft_model"):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})