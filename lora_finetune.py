import torch
import torch.nn.functional as F
from config import get_weights_file_path
from tqdm import tqdm
from torch.optim import AdamW
import warnings
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dataset import setup_dataloaders
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lora_model import get_lora_model


def get_model(config, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_classes"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    model = get_lora_model(model, **kwargs)
    return model, tokenizer


def train_model(config, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model(config, **kwargs)
    model.to(device)
    train_loader, val_loader, test_loader = setup_dataloaders(config, tokenizer)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    writer = SummaryWriter(config["experiment_name"])
    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    for epoch in range(initial_epoch, config["epochs_num"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"epoch{epoch:02d}")
        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            loss = F.cross_entropy(outputs, label, label_smoothing=0.1)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_filename = get_weights_file_path(config, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
