from pathlib import Path


def get_lora_config(**kwargs):
    """
    This function returns a dictionary with LoRa configuration options,
    using default values for missing arguments.

    defaults = {
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_query": True,
        "lora_key": True,
        "lora_value": True,
        "lora_projection": True,
        "lora_ffn": True,
        "lora_head": True,
    }

    """

    defaults = {
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_query": True,
        "lora_key": True,
        "lora_value": True,
        "lora_projection": True,
        "lora_ffn": True,
        "lora_head": True,
    }

    return {**defaults, **kwargs}


def get_config(**kwargs):
    """
    Get the configuration parameters for the model, dataset, and fine-tuning.

    Returns:
        dict: Configuration dictionary containing the parameters.


    defaults = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 20,
        "model_folder": "weights",
        "preload": None,
        "model_basename": "transformer_model",
        "experiment_name": "runs/tmodel",
        "model_name": "distilbert-base-uncased",
        "dataset_name": "dair-ai/emotion",
        "num_classes": 6,
        "lora": True,
    }

    """

    defaults = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 20,
        "model_folder": "weights",
        "preload": None,
        "model_basename": "transformer_model",
        "experiment_name": "runs/tmodel",
        "model_name": "distilbert-base-uncased",
        "dataset_name": "dair-ai/emotion",
        "num_classes": 6,
        "lora": True,
    }

    return {**defaults, **kwargs}


def get_weights_file_path(config, epoch: str):
    """
    Get the file path for saving or loading model weights for a specific epoch.

    Args:
        config (dict): Configuration dictionary containing parameters.
        epoch (str): Epoch number as a string.

    Returns:
        str: File path for the model weights file.
    """

    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}_{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
