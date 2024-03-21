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
        "lora_head": True,
        "lora_ffn": True,
        "lora_projection": False,
        "train_projection": True,
    }

    """

    defaults = {
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_query": True,
        "lora_key": True,
        "lora_value": True,
        "lora_head": True,
        "lora_ffn": True,
        "lora_projection": False,
        "train_projection": True,
    }

    lora_projection = kwargs.get(["lora_projection"], False)
    train_projection = kwargs.get(["train_projection"], True)

    assert (
        lora_projection != train_projection
        or lora_projection == train_projection is False
    ), "either to train or to apply lora or neither of them"

    return {**defaults, **kwargs}


def get_config(**kwargs):
    """
    Get the configuration parameters for the model, dataset, and fine-tuning.

    Returns:
        dict: Configuration dictionary containing the parameters.


    defaults = {
        "batch_size": 32,
        "learning_rate": 1e-5,
        "num_epochs": 20,
        "model_folder": "weights",
        "preload": 'latest',
        "model_basename": "distilbert_base_model",
        "experiment_name": "runs/tmodel",
        "model_name": "distilbert-base-uncased",
        "dataset_name": "dair-ai/emotion",
        "num_classes": 6,
        "lora_fine_tuning": True,
        "train_last_layer": True,
        "max_len": 300,
    }

    """

    defaults = {
        "batch_size": 32,
        "learning_rate": 1e-5,
        "num_epochs": 20,
        "model_folder": "weights",
        "preload": "latest",
        "model_basename": "distilbert_base_model",
        "experiment_name": "runs/tmodel",
        "model_name": "distilbert-base-uncased",
        "dataset_name": "dair-ai/emotion",
        "num_classes": 6,
        "lora_fine_tuning": True,
        "max_len": 300,
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


def latest_weights_file_path(config):
    """
    Retrieves the path to the most recently saved model weights file.

    Args:
        config (dict): A configuration dictionary containing parameters.

    Returns:
        str: The path to the latest weights file, or None if no weights files are found.
    """

    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        return None

    return str(sorted(weights_files)[-1])
