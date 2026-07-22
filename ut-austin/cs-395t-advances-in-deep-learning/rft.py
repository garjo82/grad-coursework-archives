from .base_llm import BaseLLM
from .data import Dataset
from .sft import TokenizedDataset, test_model


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_rft_example(prompt: str, answer_float: float, reasoning_chain: str) -> dict[str, str]:
    """
    Format the RFT tuple into the question/answer dict expected by TokenizedDataset.
    We supervise on the entire reasoning chain, not just the float.
    """
    return {
        "question": prompt,
        "answer": reasoning_chain
    }


def train_model(
        output_dir: str = "homework/rft_model",
        **kwargs,
):
    from peft import PeftModel
    from transformers import Trainer, TrainingArguments

    # 1. Load the synthetic dataset
    try:
        train_data = Dataset("rft")
    except FileNotFoundError:
        raise FileNotFoundError("data/rft.json not found. You must run datagen.py first.")

    llm = BaseLLM()

    # 2. Apply the RFT formatting
    train_dataset = TokenizedDataset(llm.tokenizer, train_data, format_rft_example)

    # 3. CRITICAL: Warm-start from the r=10 SFT adapter weights
    print("Loading SFT adapter weights for RFT warm-start...")
    model = PeftModel.from_pretrained(
        llm.model,
        "homework/sft_model",
        is_trainable=True
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 4. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # Halved to safely fine-tune the CoT without wrecking the SFT math
        num_train_epochs=5,  # 5 epochs is plenty when inheriting weights
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

    # 6. Train and save the finalized RFT adapter
    trainer.train()
    trainer.save_model(output_dir)

    # Run evaluation natively
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})