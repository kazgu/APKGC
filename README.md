# APKGC: an Adaptive and Prompt-Tuning Framework for Knowledge Graph Completion

This repository contains a Knowledge Graph Completion (KGC) model that combines traditional KG embeddings with fine-tuned Large Language Models. The model can predict missing entities in knowledge graph triplets by leveraging both structured KG embeddings and the language understanding capabilities of LLMs.

## Overview

The model uses a fine-tuned LLaMA-2 model enhanced with Knowledge Graph Embeddings to complete partial knowledge graph triplets. It employs parameter-efficient fine-tuning (PEFT) with LoRA adapters, enabling training on consumer-grade GPUs.

## Refactored Codebase

The codebase has been refactored with the following improvements:

1. **Modular Architecture**: Separated the code into distinct modules with clear responsibilities:
   - `config.py`: Configuration management using dataclasses
   - `data_utils.py`: Dataset loading and processing
   - `model.py`: Model architecture implementations
   - `train.py`: Training pipeline

2. **Improved Attention Adapter**: The model uses an attention-based adapter to transform knowledge graph embeddings into the language model's embedding space, replacing the simpler feed-forward network in the original implementation.

3. **Better Prompt Engineering**: Enhanced prompt templates for knowledge graph completion tasks.

4. **Optimized Training Parameters**: Improved LoRA configuration and training hyperparameters.

5. **Comprehensive Documentation**: Added docstrings and comments to explain the purpose and functionality of each component.

6. **Error Handling and Logging**: Added proper logging and error handling throughout the codebase.

## Model Architecture

The architecture consists of three main components:

1. **Base Language Model**: A quantized LLaMA-2 model fine-tuned with LoRA adapter layers.

2. **Knowledge Graph Embedding Model**: Transforms pre-trained entity and relation embeddings into the language model's embedding space using a multi-head attention adapter.

3. **KGELlama**: Combines the language model and KG embeddings by replacing special token embeddings with the corresponding KG embeddings during forward passes.

## Usage

### Configuration

The model configuration is centralized in `config.py`. You can modify the default configuration by editing this file or by extending the `Config` class.

### Training

To train the model:

```bash
python train.py
```

### Customization

You can customize the model by modifying the configuration in `config.py`. Key parameters include:

- `model.model_name`: Path to the pre-trained language model
- `data.data_path`: Path to the knowledge graph dataset
- `data.kge_embedding_dir`: Path to pre-trained KG embeddings
- `model.lora_r`: LoRA rank for parameter-efficient fine-tuning
- `model.lora_alpha`: LoRA alpha parameter
- `training.*`: Various training hyperparameters

## Data Format

The model expects a dataset in a format compatible with the Hugging Face `datasets` library, with the following fields:

- `input`: The input prompt containing an incomplete triplet
- `output`: The expected completion
- `query_id`: ID for the relation query
- `entity_ids`: IDs for the entity candidates
- `topk_names`: Names of the top-k candidate entities

## Requirements

- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- BitsAndBytes (for quantization)
