import torch
import torch.nn as nn
from config import get_lora_config


class LoRALayer(nn.Module):
    """
    Implements the Low-Rank Adaptive Linear Transformation (LoRA) layer.

    This layer performs a linear transformation with learnable low-rank factors,
    reducing computational cost and potentially improving model performance.

    Args:
        fan_in: Number of input features (int).
        fan_out: Number of output features (int).
        rank: Rank of the low-rank factors (int).
        alpha: Hyperparameter scaling the output (int).

    Attributes:
        alpha: Hyperparameter scaling the output (int).
        A: Low-rank factor matrix of shape (fan_in, rank) (float tensor).
        B: Low-rank factor matrix of shape (rank, fan_out) (float tensor).

    Inputs:
        x: Input tensor of shape (..., fan_in) (float tensor).

    Outputs:
        Transformed tensor of shape (..., fan_out) (float tensor).
    """

    def __init__(self, fan_in: int, fan_out: int, rank: int, alpha: int):
        super().__init__()
        self.alpha = alpha
        std = torch.tensor(rank, dtype=torch.float) ** -0.5
        self.A = nn.Parameter(torch.randn(fan_in, rank) * std)
        self.B = nn.Parameter(torch.zeros(rank, fan_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """
    Combines a linear layer with a LoRALayer to perform a combined transformation.

    This module wraps a linear layer and adds a LoRALayer in parallel. The
    output is the sum of the linear layer's output and the LoRALayer's output.

    Args:
        linear_layer: The original linear layer to be wrapped (nn.Module).
        rank: Rank of the low-rank factors in the LoRALayer (int).
        alpha: Hyperparameter scaling the LoRALayer output (int).

    Attributes:
        linear_layer: Original linear layer (frozen, nn.Module).
        alpha: Hyperparameter scaling the LoRALayer output (int).
        lora_layer: LoRALayer instance (nn.Module).

    Inputs:
        x: Input tensor of shape (..., in_features) (float tensor).

    Outputs:
        Transformed tensor of shape (..., out_features) (float tensor).
    """

    def __init__(self, linear_layer: nn.Module, rank: int, alpha: int):
        super().__init__()
        self.linear_layer = linear_layer
        self.alpha = alpha
        self.lora_layer = LoRALayer(
            linear_layer.in_features, linear_layer.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear_layer(x) + self.lora_layer(x)


def get_lora_model(model: nn.Module, config: dict = None, **kwargs) -> nn.Module:
    """
    Applies LoRA (Low-Rank Adaptive Linear Transformation) to a pre-trained model.

    This function selectively replaces linear layers within a model with
    corresponding LoRALayer instances based on a configuration dictionary.
    The configuration specifies which parts of the model to apply LoRA to
    and hyperparameters like rank and alpha.

    Args:
        model: The pre-trained model to be modified (nn.Module).
        config: Optional dictionary containing LoRA configuration (default: None).
                If not provided, a default configuration is retrieved using
                `get_lora_config` (kwargs are passed to this function).
        **kwargs: Additional keyword arguments passed to `get_lora_config`.

    Returns:
        The modified model with LoRA layers applied (nn.Module).
    """

    # Ensure config is available
    if config is None:
        config = get_lora_config(**kwargs)

    # Freeze the pre-trained model parameters
    model.requires_grad_(False)

    def assign_lora(linear_layer):
        return LinearWithLoRA(
            linear_layer, rank=config["lora_rank"], alpha=config["lora_alpha"]
        )

    for layer in model.distilbert.transformer.layer:
        if config["lora_query"]:
            layer.attention.q_lin = assign_lora(layer.attention.q_lin)
        if config["lora_key"]:
            layer.attention.k_lin = assign_lora(layer.attention.k_lin)
        if config["lora_value"]:
            layer.attention.v_lin = assign_lora(layer.attention.v_lin)
        if config["lora_projection"]:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin)
        if config["lora_ffn"]:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2)

    if config["lora_projection"]:
        model.pre_classifier = assign_lora(model.pre_classifier)
        model.classifier = assign_lora(model.classifier)
    elif config["train_projection"]:
        model.pre_classifier.requires_grad_(True)
        model.classifier.requires_grad_(True)

    return model
