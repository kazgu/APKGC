"""Configuration management for knowledge graph completion model."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_name: str = "~/.cache/huggingface/hub/models--daryl149--llama-2-7b-chat-hf"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    lora_r: int = 16
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    special_tokens: List[str] = field(default_factory=lambda: ['[QUERY]', '[ENTITY]', '[RELATION]'])


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_path: str = 'dataset/WN18RR/SimKGC/data_KGELLM/'
    kge_embedding_dir: str = 'dataset/WN18RR/SimKGC/'
    embedding_input_size: int = 768
    embedding_intermediate_size: int = 1024


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./results"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    optim: str = "paged_adamw_32bit"
    num_train_epochs: int = 5
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: str = "none"
    fp16: bool = True
    warmup_ratio: float = 0.03
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    source_max_len: int = 1024
    target_max_len: int = 64
    final_model_output_dir: str = "./final_model"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "auto"
    cuda_visible_devices: str = "0"
    seed: int = 42


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
