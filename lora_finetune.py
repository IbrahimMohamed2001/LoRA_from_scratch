import torch
import torch.nn.functional as F
from config import get_weights_file_path, get_config, latest_weights_file_path
from tqdm import tqdm
from torch.optim import AdamW
import warnings
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from dataset import setup_dataloaders
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lora_model import get_lora_model
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score


def get_model(config, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=config["num_classes"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if config["lora"]:
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
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
    )

    if model_filename:
        print(f"preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = config["learning_rate"]
        global_step = state["global_step"]

    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": Accuracy(
                task="multiclass",
                num_classes=config["num_classes"],
                average="macro",
            ),
            "precision": Precision(
                task="multiclass",
                num_classes=config["num_classes"],
                average="weighted",
            ),
            "recall": Recall(
                task="multiclass",
                num_classes=config["num_classes"],
                average="weighted",
            ),
            "f1_score": F1Score(
                task="multiclass",
                num_classes=config["num_classes"],
                average="weighted",
            ),
        }
    )
    metric_tracker = torchmetrics.MetricTracker(metrics).to(device)
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    print(
        "trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"epoch {epoch:02d}")

        avg_train_loss = 0.0

        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask).logits

            loss = F.cross_entropy(
                outputs, label, ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
            )
            avg_train_loss += loss.item()

            metric_tracker.increment()

            predictions = torch.argmax(outputs, dim=1)
            metric_tracker.update(predictions, label)

            batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        avg_train_loss /= len(batch_iterator)
        results = metric_tracker.compute()
        train_acc = results["accuracy"].item()
        train_prec = results["precision"].item()
        train_recall = results["recall"].item()
        train_f1 = results["f1_score"].item()

        metric_tracker.reset()

        writer.add_scalar("train_average_loss", avg_train_loss, epoch)
        writer.add_scalar("train_accuracy", train_acc, epoch)
        writer.add_scalar("train_precision", train_prec, epoch)
        writer.add_scalar("train_recall", train_recall, epoch)
        writer.add_scalar("train_f1", train_f1, epoch)
        writer.flush()

        validate_model(model, val_loader, config, device, writer, epoch)

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


def validate_model(model, validation_loader, config, device, writer, epoch):
    model.eval()

    with torch.no_grad():
        batch_iterator = tqdm(validation_loader, desc=f"epoch {epoch:02d}")

        avg_val_loss = 0.0
        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": Accuracy(
                    task="multiclass",
                    num_classes=config["num_classes"],
                    average="macro",
                ),
                "precision": Precision(
                    task="multiclass",
                    num_classes=config["num_classes"],
                    average="weighted",
                ),
                "recall": Recall(
                    task="multiclass",
                    num_classes=config["num_classes"],
                    average="weighted",
                ),
                "f1_score": F1Score(
                    task="multiclass",
                    num_classes=config["num_classes"],
                    average="weighted",
                ),
            }
        )
        metric_tracker = torchmetrics.MetricTracker(metrics).to(device)

        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask).logits

            val_loss = F.cross_entropy(outputs, label)
            avg_val_loss += val_loss

            predictions = torch.argmax(outputs, dim=1)
            metric_tracker.increment()
            metric_tracker.update(predictions, label)

            batch_iterator.set_postfix({"val_loss": f"{val_loss.item():6.3f}"})

        results = metric_tracker.compute()
        avg_val_loss /= len(batch_iterator)

        val_acc = results["accuracy"].item()
        val_precision = results["precision"].item()
        val_recall = results["recall"].item()
        val_f1 = results["f1_score"].item()

        writer.add_scalar("val_average_loss", avg_val_loss, epoch)
        writer.add_scalar("val_accuracy", val_acc, epoch)
        writer.add_scalar("val_precision", val_precision, epoch)
        writer.add_scalar("val_recall", val_recall, epoch)
        writer.add_scalar("val_f1", val_f1, epoch)
        writer.flush()

        metric_tracker.reset()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
