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
