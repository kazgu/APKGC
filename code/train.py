#!/usr/bin/env python
# coding: utf-8
"""Training script for the Knowledge Graph Embedding enhanced Language Model."""

import os
import sys
import logging
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

from config import get_default_config
from data_utils import get_chatml_dataset, KGDataCollator
from model import EmbeddingModel, KGELLM
from tensor_utils import convert_tensor_to_python


def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.training.logging_dir, "training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Configuration: {config}")


def set_environment(config):
    """Set environment variables and random seeds for reproducibility."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    
    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    set_seed(config.seed)
    
    # Create output directories if they don't exist
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.logging_dir, exist_ok=True)
    os.makedirs(config.training.final_model_output_dir, exist_ok=True)


def load_tokenizer(config):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token
    
    # Add special tokens for KG entities and relations
    tokenizer.add_tokens(config.model.special_tokens)
    
    return tokenizer


def load_base_model(config):
    """Load the base language model with quantization."""
    # Configure quantization for memory efficiency
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.model.load_in_4bit,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load quantized base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quant_config,
        device_map=config.device
    )
    
    return base_model


def add_lora_adapter(base_model, config, tokenizer):
    """Add LoRA adapter to the base model for parameter-efficient fine-tuning."""
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias=config.model.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=config.model.target_modules
    )
    
    # Prepare model for k-bit training and add LoRA adapter
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, lora_config)
    
    
    return model


def create_kg_enhanced_model(tokenizer, lora_model, config):
    """Create the Knowledge Graph enhanced language model."""
    # Load KG embedding model with contrastive learning support
    embed_model = EmbeddingModel(
        config.data.kge_embedding_dir, 
        config.data.embedding_input_size, 
        config.data.embedding_intermediate_size, 
        lora_model.config.hidden_size, 
        lora_model.config.hidden_act,
        contrastive_dim=config.model.contrastive_dim if hasattr(config.model, 'contrastive_dim') else config.training.contrastive_dim if hasattr(config.training, 'contrastive_dim') else 512
    )
    
    # Create KG-enhanced language model
    kg_model = KGELLM(tokenizer, lora_model, embed_model)
    
    return kg_model


def prepare_training_args(config):
    """Prepare training arguments."""
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        optim=config.training.optim,
        num_train_epochs=config.training.num_train_epochs,
        logging_dir=config.training.logging_dir,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        report_to=config.training.report_to,
        fp16=config.training.fp16,
        remove_unused_columns=config.training.remove_unused_columns,
        warmup_ratio=config.training.warmup_ratio,
        dataloader_num_workers=config.training.dataloader_num_workers,
    )
    
    return training_args


class KGCompletionTrainer(Trainer):
    """Custom trainer for Knowledge Graph Completion using Language Models with contrastive learning."""
    
    def __init__(self, 
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 kg_metrics=True,
                 contrastive_weight=0.5):
        """Initialize KGCompletionTrainer.
        
        Args:
            model: Model to train
            args: Training arguments
            data_collator: Function to collate batch data
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            model_init: Function to initialize model
            compute_metrics: Function to compute metrics
            callbacks: List of callbacks
            optimizers: Tuple of (optimizer, scheduler)
            preprocess_logits_for_metrics: Function to preprocess logits for metrics
            kg_metrics: Whether to use KG-specific metrics
            contrastive_weight: Weight for contrastive loss
        """
        # Initialize contrastive weight with warmup
        self.contrastive_weight = contrastive_weight
        self.contrastive_warmup_steps = 100  # Gradually increase contrastive weight
        self.current_step = 0
            
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
    def log(self, logs):
        """Override default log method to convert PyTorch tensors to Python types.
        
        Args:
            logs: Dictionary of logging information
        """
        # Convert any tensors to Python types
        logs = convert_tensor_to_python(logs)
        
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute training loss with contrastive learning component.
        
        Args:
            model: Model to train
            inputs: Dictionary of input tensors
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss value or tuple of (loss, outputs)
        """
        # Extract inputs for language modeling
        lm_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['labels'],
            'query_ids': inputs['query_ids'],
            'entity_ids': inputs['entity_ids']
        }
        
        # Forward pass for language modeling
        outputs = model(**lm_inputs)
        lm_loss = outputs.loss
        
        # Compute contrastive loss if positive_pairs is provided
        contrastive_loss = torch.tensor(0.0, device=lm_loss.device)
        if 'positive_pairs' in inputs:
            # Get query and entity embeddings with projections for contrastive learning
            query_embeds, entity_embeds, query_proj, entity_proj = model.kge_model(
                inputs['query_ids'], 
                inputs['entity_ids'].view(-1),
                return_projections=True
            )
            
            # Check if hard negatives are available
            hard_negative_ids = inputs.get('hard_negative_ids', None)
            hard_negative_mask = inputs.get('hard_negative_mask', None)
            
            # Compute contrastive loss with hard negatives if available
            contrastive_loss = model.kge_model.compute_contrastive_loss(
                query_proj, 
                entity_proj, 
                inputs['positive_pairs'],
                hard_negative_ids,
                hard_negative_mask
            )
            
            # Log contrastive loss and similarity statistics
            with torch.no_grad():
                # Calculate similarity statistics for monitoring
                similarity = torch.matmul(query_proj, entity_proj.transpose(0, 1))
                positive_mask = inputs['positive_pairs'].float()
                
                # Get positive and negative similarities
                positive_sim = (similarity * positive_mask).sum() / positive_mask.sum().clamp(min=1)
                negative_mask = 1 - positive_mask
                negative_sim = (similarity * negative_mask).sum() / negative_mask.sum().clamp(min=1)
                
                # Calculate max similarity for each query with negative entities
                max_neg_sim_per_query = []
                for i in range(query_proj.size(0)):
                    neg_mask = negative_mask[i]
                    if neg_mask.sum() > 0:
                        neg_sims = similarity[i] * neg_mask
                        neg_sims[neg_mask == 0] = float('-inf')  # Mask out positive pairs
                        max_neg_sim = neg_sims.max().item()
                        max_neg_sim_per_query.append(max_neg_sim)
                
                max_neg_sim = sum(max_neg_sim_per_query) / len(max_neg_sim_per_query) if max_neg_sim_per_query else 0
                
                # Log detailed metrics - values will be converted to Python types by our overridden log method
                self.log({
                    "contrastive_loss": contrastive_loss.detach(),
                    "positive_similarity": positive_sim.detach(),
                    "negative_similarity": negative_sim.detach(),
                    "max_negative_similarity": max_neg_sim,
                    "similarity_gap": (positive_sim - negative_sim).detach(),
                    "min_similarity_gap": (positive_sim - max_neg_sim)
                })
        
        # Apply warmup to contrastive weight with a longer warmup period
        self.current_step += 1
        effective_weight = self.contrastive_weight
        
        # Use a longer warmup period (500 steps) with a gradual increase
        if self.current_step < 500:
            # Gradually increase the weight using a smoother curve
            progress = self.current_step / 500
            effective_weight = self.contrastive_weight * (progress ** 2)  # Quadratic warmup
        
        # Apply curriculum learning for contrastive loss
        # Start with easier examples (larger margin) and gradually make it harder
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: Contrastive weight = {effective_weight:.4f}")
        
        # Combine losses with effective weight
        # Use a dynamic weighting scheme based on the relative magnitudes of the losses
        # This helps prevent one loss from dominating the other
        lm_loss_scale = 1.0
        contrastive_loss_scale = effective_weight
        
        # Normalize the losses to have similar scales
        if self.current_step > 10:  # After a few steps to gather statistics
            # Use a moving average to stabilize the scaling
            if not hasattr(self, 'lm_loss_avg'):
                self.lm_loss_avg = lm_loss.item()
                self.contrastive_loss_avg = contrastive_loss.item() if contrastive_loss.item() > 0 else 1.0
            else:
                # Update moving averages
                self.lm_loss_avg = 0.9 * self.lm_loss_avg + 0.1 * lm_loss.item()
                if contrastive_loss.item() > 0:
                    self.contrastive_loss_avg = 0.9 * self.contrastive_loss_avg + 0.1 * contrastive_loss.item()
            
            # Adjust scales to make the weighted losses comparable
            if self.contrastive_loss_avg > 0:
                ratio = self.lm_loss_avg / self.contrastive_loss_avg
                contrastive_loss_scale = effective_weight * min(max(ratio, 0.1), 10.0)
        
        # Combine the losses
        total_loss = lm_loss_scale * lm_loss + contrastive_loss_scale * contrastive_loss
        
        # Log individual losses and weights
        self.log({
            "lm_loss": lm_loss.detach(),
            "lm_loss_scale": lm_loss_scale,
            "contrastive_loss_raw": contrastive_loss.detach(),
            "contrastive_loss_scale": contrastive_loss_scale,
            "weighted_contrastive_loss": (contrastive_loss_scale * contrastive_loss).detach(),
            "total_loss": total_loss.detach(),
            "step": self.current_step
        })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs):
        """Perform a training step on a batch of inputs.
        
        Args:
            model: Model to train
            inputs: Dictionary of input tensors
        
        Returns:
            Loss value
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss with contrastive component
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Handle fp16 training properly through the accelerator
        # This ensures proper gradient scaling
        self.accelerator.backward(loss)
        
        return loss.detach()
    


def train(config):
    """Main training function."""
    # Set up environment
    set_environment(config)
    
    # Load and configure tokenizer
    tokenizer = load_tokenizer(config)
    
    # # Load base model
    base_model = load_base_model(config)
    
    # # Add LoRA adapter
    lora_model = add_lora_adapter(base_model, config, tokenizer)
    
    # Load dataset (with or without hard negatives)
    if config.data.use_hard_negatives and os.path.exists(config.data.data_path_with_hard_negatives):
        print(f"Using dataset with hard negatives from: {config.data.data_path_with_hard_negatives}")
        dataset = get_chatml_dataset(config.data.data_path_with_hard_negatives)
    else:
        print(f"Using standard dataset from: {config.data.data_path}")
        dataset = get_chatml_dataset(config.data.data_path)
    
    # # Create KG-enhanced model
    kg_model = create_kg_enhanced_model(tokenizer, lora_model, config)
    
    # # Prepare training arguments
    training_args = prepare_training_args(config)
    
    # Initialize data collator with contrastive learning and hard negative mining support
    data_collator = KGDataCollator(
        tokenizer,
        config.training.source_max_len,
        config.training.target_max_len,
        contrastive_learning=config.training.contrastive_learning,
        use_hard_negatives=True
    )


    # Initialize trainer with contrastive learning
    trainer = KGCompletionTrainer(
        model=kg_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else None,
        data_collator=data_collator,
        contrastive_weight=config.training.contrastive_weight
    )
    
    # Start training
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training completed successfully!")
    except KeyboardInterrupt:
        logging.info('Exiting from training early due to interruption')
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise
    
    # Save final model
    logging.info(f"Saving final model to {config.training.final_model_output_dir}")
    kg_model.save_pretrained(config.training.final_model_output_dir)
    logging.info("Model saved successfully!")


if __name__ == "__main__":
    # Load configuration
    config = get_default_config()
    
    # Setup logging
    os.makedirs(config.training.logging_dir, exist_ok=True)
    setup_logging(config)
    
    # Run training
    train(config)
    
    logging.info("Done!")
