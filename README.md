# LoRA Fine-Tuning for Sequence Classification

This repository contains code for fine-tuning a sequence classification model using Low-Rank Adaptation (LoRA). The implementation leverages the Hugging Face Transformers library and PyTorch for model training and evaluation.

## Table of Contents

- [LoRA Fine-Tuning for Sequence Classification](#lora-fine-tuning-for-sequence-classification)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Configuration](#configuration-1)
  - [Training](#training-1)
  - [Evaluation](#evaluation-1)
  - [Acknowledgements](#acknowledgements)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

The configuration for the model, dataset, and fine-tuning parameters is defined in `config.py`. You can modify the default values or pass custom values when calling the functions.

### Training

To train the model, run:

```bash
python lora_finetune.py
```

This will start the training process using the configuration specified in `config.py`.

### Evaluation

The model is evaluated on the validation set during training. You can also evaluate the model separately by calling the `validate_model` function with the appropriate parameters.

## Configuration

The configuration parameters are defined in `config.py`. Key parameters include:

- `batch_size`: Batch size for training and evaluation.
- `learning_rate`: Learning rate for the optimizer.
- `num_epochs`: Number of training epochs.
- `model_folder`: Directory to save model weights.
- `preload`: Preload option for model weights (`latest` or specific epoch).
- `model_name`: Pre-trained model name from Hugging Face.
- `dataset_name`: Dataset name from Hugging Face Datasets.
- `num_classes`: Number of classes for classification.
- `lora_fine_tuning`: Boolean to enable/disable LoRA fine-tuning.
- `max_len`: Maximum sequence length for tokenization.

## Training

The training process involves the following steps:

1. Load the configuration and initialize the model and tokenizer.
2. Set up the data loaders for training, validation, and test sets.
3. Train the model using the specified optimizer and learning rate.
4. Save the model weights after each epoch.

## Evaluation

The evaluation process involves computing metrics such as accuracy, precision, recall, and F1 score on the validation set. These metrics are logged using TensorBoard.

## Acknowledgements

This project uses the following libraries:

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

Feel free to modify this `README.md` file to better suit your project's needs.
