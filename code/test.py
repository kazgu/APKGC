#!/usr/bin/env python
# coding: utf-8
"""
Test script for Knowledge Graph Embedding enhanced Language Model.

This script evaluates the performance of a fine-tuned language model
on knowledge graph completion tasks.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import get_default_config, Config
from data_utils import get_chatml_dataset
from model import EmbeddingModel, KGELLM


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test Knowledge Graph Completion model")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default=None,
        help="Path to the base model"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="final_model",
        help="Path to the adapter model"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["WN18RR", "FB15K237"], 
        default="WN18RR",
        help="Dataset to use for testing"
    )
    parser.add_argument(
        "--kge_method", 
        type=str, 
        choices=["SimKGC", "TransE", "CoLE"], 
        default="SimKGC",
        help="KGE method to use"
    )
    parser.add_argument(
        "--gpu", 
        type=str, 
        default="0",
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        default=True,
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def set_environment(gpu: str) -> None:
    """Set environment variables.
    
    Args:
        gpu: GPU device ID
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    logger.info(f"Using GPU device: {gpu}")


def load_model_and_tokenizer(
    config: Config, 
    base_model_path: Optional[str], 
    adapter_path: str
) -> Tuple[AutoTokenizer, PeftModel]:
    """Load the model and tokenizer.
    
    Args:
        config: Configuration object
        base_model_path: Path to the base model
        adapter_path: Path to the adapter model
        
    Returns:
        Tuple of (tokenizer, model)
    """
    # Use config model path if base_model_path is not provided
    model_path = base_model_path if base_model_path else config.model.model_name
    logger.info(f"Loading base model from {model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=config.model.load_in_4bit,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(config.model.special_tokens)
    
    # load adapter
    logger.info(f"Loading LoRA adapters from {adapter_path}")

    model = PeftModel.from_pretrained(base_model, adapter_path) 
    
    return tokenizer, model
 

def load_kg_model(
    config: Config, 
    tokenizer: AutoTokenizer, 
    model: PeftModel, 
    dataset: str, 
    kge_method: str, 
    adapter_path: str
) -> Tuple[KGELLM, str]:
    """Load the KG-enhanced model.
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer
        model: Base model with adapter
        dataset: Dataset name
        kge_method: KGE method name
        adapter_path: Path to the adapter model
        
    Returns:
        Tuple of (KG-enhanced model, data path)
    """
    # Set KGE embedding directory and data path
    kge_embedding_dir = f"dataset/{dataset}/{kge_method}"
    data_path = f"{kge_embedding_dir}/data_KGELLM/"
    logger.info(f"Using KGE embeddings from {kge_embedding_dir}")
    
    # Load embedding model
    embed_model = EmbeddingModel(
        kge_embedding_dir, 
        config.data.embedding_input_size, 
        config.data.embedding_intermediate_size, 
        model.config.hidden_size, 
        model.config.hidden_act
    ).cuda()
    
    # Load KGE checkpoint
    kge_checkpoint = torch.load(os.path.join(adapter_path, 'kge.bin'))
    embed_model.load_state_dict(kge_checkpoint)
    
    # Create KG-enhanced model
    kg_model = KGELLM(tokenizer, model, embed_model).cuda()
    kg_model.eval()
    
    return kg_model, data_path


def compute_metrics(ranks: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        ranks: Array of ranks
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        f'{prefix}hits1': np.mean(ranks <= 1),
        f'{prefix}hits3': np.mean(ranks <= 3),
        f'{prefix}hits10': np.mean(ranks <= 10),
        f'{prefix}mrr': np.mean(1. / ranks),
    }
    metrics = {k: round(v, 3) for k, v in metrics.items()}
    logger.info(f"Samples: {ranks.shape[0]}; Metrics: {metrics}")
    return metrics


def evaluate_model(
    kg_model: KGELLM, 
    tokenizer: AutoTokenizer, 
    dataset: Dict, 
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate the model on the test dataset.
    
    Args:
        kg_model: KG-enhanced model
        tokenizer: Tokenizer
        dataset: Dataset dictionary
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize arrays for storing results
    raw_ranks = np.array([])
    ranks = np.array([])
    predictions = []
    correct_count = 0
    total_count = 0
    
    # Evaluate on test dataset
    logger.info("Starting evaluation on test dataset")
    for example in tqdm(dataset['test']):
        # Prepare inputs
        messages = example['input']
        ground_truth = example['output']
        
        # Tokenize input
        inputs = tokenizer(
            messages,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Prepare query and entity IDs
        query_ids = torch.LongTensor([example['query_id']]).cuda()
        entity_ids = torch.LongTensor([example['entity_ids']]).cuda()
        
        # Generate prediction
        with torch.inference_mode():
            generation_output = kg_model.generate(
                input_ids=inputs['input_ids'],
                query_ids=query_ids,
                entity_ids=entity_ids
            )
        
        # Decode prediction
        prediction = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
        # Print prediction and ground truth if verbose
        if verbose:
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Ground truth: {ground_truth}")
            logger.info("=" * 40)
        
        # Update correct count
        is_correct = ground_truth in prediction
        correct_count += int(is_correct)
        total_count += 1
        accuracy = correct_count / total_count
        
        if verbose:
            logger.info(f"Current accuracy: {accuracy:.3f}")
        
        # Calculate rank
        rank = example['rank']
        topk_names = example['topk_names']
        
        if ground_truth == prediction.strip():
            rank = 1
        elif prediction.strip() not in set(topk_names) or topk_names.index(prediction.strip()) >= rank:
            rank += 1
        
        # Store results
        example_result = {
            'target': ground_truth,
            'pred_rank': rank,
            'pred': prediction.strip()
        }
        predictions.append(example_result)
        raw_ranks = np.append(raw_ranks, example['rank'])
        ranks = np.append(ranks, rank)
        
        # Compute and display metrics periodically
        if total_count % 10 == 0 or total_count == len(dataset['test']):
            logger.info(f"Processed {total_count}/{len(dataset['test'])} examples")
            logger.info("Raw metrics:")
            compute_metrics(raw_ranks, "raw_")
            logger.info("Prediction metrics:")
            compute_metrics(ranks, "pred_")
    
    # Compute final metrics
    final_metrics = {
        "accuracy": accuracy,
        **compute_metrics(raw_ranks, "raw_"),
        **compute_metrics(ranks, "pred_")
    }
    
    return final_metrics


def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set environment
    set_environment(args.gpu)
    
    # Load configuration
    config = get_default_config()
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(
        config, 
        args.base_model_path, 
        args.adapter_path
    )
    
    # Load KG model
    kg_model, data_path = load_kg_model(
        config, 
        tokenizer, 
        model, 
        args.dataset, 
        args.kge_method, 
        args.adapter_path
    )
    
    # Load dataset
    dataset = get_chatml_dataset(data_path)
    
    # Evaluate model
    metrics = evaluate_model(kg_model, tokenizer, dataset, args.verbose)
    
    # Print final results
    logger.info("=" * 80)
    logger.info("Final evaluation results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.3f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()