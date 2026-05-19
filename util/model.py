from __future__ import annotations


def move_to_model_device(inputs: dict, model) -> dict:
    import torch

    return {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def first_token_id(token_id) -> int | None:
    if isinstance(token_id, (list, tuple)):
        return token_id[0] if token_id else None
    return token_id


def configure_generation_tokens(model, processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = first_token_id(eos_token_id)

    if pad_token_id is None:
        pad_token_id = first_token_id(getattr(model.generation_config, "eos_token_id", None))

    if pad_token_id is not None:
        model.generation_config.pad_token_id = pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_token_id

    return pad_token_id
