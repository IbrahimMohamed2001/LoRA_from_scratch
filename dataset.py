import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class EmotionDataset(Dataset):
    def __init__(self, dataset, tokenizer, partition_key="train") -> None:
        super().__init__()
        self.partition = dataset[partition_key]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.partition)

    def __getitem__(self, index):
        text = self.partition[index]["text"]
        label = torch.tensor(self.partition[index]["label"], dtype=torch.int8)

        tokenized = self.tokenizer(text, truncation=True, padding=True)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.int64)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.int8)

        return {
            "text": text,
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def setup_dataloaders(config, tokenizer):
    dataset = load_dataset(config["dataset_name"])
    train_dataset = EmotionDataset(dataset, tokenizer, partition_key="train")
    val_dataset = EmotionDataset(dataset, tokenizer, partition_key="validation")
    test_dataset = EmotionDataset(dataset, tokenizer, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"])
    return train_loader, val_loader, test_loader
