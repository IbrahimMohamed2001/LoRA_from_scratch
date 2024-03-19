import torch
import torch.nn.functional as F
from config import get_weights_file_path, get_config
from tqdm import tqdm
from torch.optim import AdamW
import warnings
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dataset import setup_dataloaders
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lora_model import get_lora_model
from torchmetrics import Accuracy, Precision, Recall, F1Score


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
    train_loader, val_loader, _ = setup_dataloaders(config, tokenizer)

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

        avg_loss = 0.0

        accuracy_metric = Accuracy(multiclass=True).to(device)
        precision_metric = Precision(multiclass=True, average="weighted").to(device)
        recall_metric = Recall(multiclass=True, average="weighted").to(device)
        f1_metric = F1Score(multiclass=True, average="weighted").to(device)

        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            loss = F.cross_entropy(outputs, label, label_smoothing=0.1)
            avg_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            accuracy_metric.update(predictions, label)
            precision_metric.update(predictions, label)
            recall_metric.update(predictions, label)
            f1_metric.update(predictions, label)

            batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss /= len(batch_iterator)
        train_acc = accuracy_metric.compute().item()
        train_prec = precision_metric.compute().item()
        train_recall = recall_metric.compute().item()
        train_f1 = f1_metric.compute().item()

        writer.add_scalar("train_accuracy", train_acc, epoch)
        writer.add_scalar("train_precision", train_prec, epoch)
        writer.add_scalar("train_recall", train_recall, epoch)
        writer.add_scalar("train_f1", train_f1, epoch)
        writer.flush()

        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
