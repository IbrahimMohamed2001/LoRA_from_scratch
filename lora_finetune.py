import torch
import torch.nn.functional as F
from config import get_lora_config
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


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model(config)
    train_loader, val_loader, test_loader = setup_dataloaders(config, tokenizer)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-9)
    writer = SummaryWriter(config["experiment_name"])
    initial_epoch = 0
    global_step = 0

    for epoch in range(config["epochs_num"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"epoch{epoch:02d}")
        for batch in batch_iterator:
            input_ids = batch["input_ids"]
            label = batch["label"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids, attention_mask)

            loss = F.cross_entropy(outputs, label, label_smoothing=0.1)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
