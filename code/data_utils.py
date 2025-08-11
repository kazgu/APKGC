"""Utilities for loading and processing knowledge graph completion datasets."""

import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import transformers

# Constant for masking tokens in the loss calculation
IGNORE_INDEX = -100



def get_part(key,tt):
    ff=False
    for t in tt.split('\n'):
        if ff:
            return t
        if key in t:
            ff=True

def transform_kg_example(example):
    prompt='''
    def entity_completion_task():
        """
        Entity completion task: Predict the missing head entity in a triplet.
        
        Given:
        - A triplet with missing head entity: #query
        - Entity description for context
        - Supporting triples for knowledge context
        - List of candidate entities
        
        Returns:
        - The most likely entity for 'h'
        """
        
        # Problem definition
        target_triplet = #target_triplet
        
        # Entity description for context
        entity_descriptions = {
            "protoctist family": #entity_descriptions
        }
        
        # Supporting triples for knowledge context
        supporting_triples = 
            #supporting_triples
        
        
        # Candidate entities
        candidate_entities = 
        #candidate_entities
        
        
        # Task: Determine the most likely entity for 'h'
        
        return "predicted_entity"
    '''


    """Transform a knowledge graph example by creating a prompt with candidates.
    
    Args:
        example: Dictionary containing the KG example data
        
    Returns:
        Dictionary with transformed input and output fields
    """
    # Extract triplet from the input
    triplet = example['input'].split('\n')[0].split('unknown:')[1][:-1].strip()
    
    # Format entity candidates
    entities = [f"{entity} [ENITITY] " for entity in example['topk_names']]

    supporting_triples=get_part('Following are some triplets about',example['input'])
    entity_descriptions=get_part('Following are some details about',example['input'])

    instruction=prompt.replace('#query',f'{triplet}')
    instruction=instruction.replace('#target_triplet',triplet.replace('[QUERY]',''))
    instruction=instruction.replace('#entity_descriptions',f'{entity_descriptions}')
    instruction=instruction.replace('#supporting_triples',f'{supporting_triples}')
    instruction=instruction.replace('#candidate_entities',f'{entities}')

    # Create instruction with triplet and candidates
    # instruction = (
    #     f"There are a incomplete triplet({triplet}), please complete it. "
    #     f"The unknown entity can choose from the below candidate entities.\n"
    #     f"Candidate entities: {entities}\n[ANSWER]: "
    # )
    
    # Find hard negative samples
    # Hard negatives are entities that are semantically similar to the correct entity
    # but are incorrect for the given triplet
    correct_entity = example['output']
    
    # Get candidate entities (excluding the correct one)
    candidate_entities = [e for e in example['topk_names'] if e != correct_entity]
    
    # Select hard negative samples (entities ranked highly by the model)
    # We'll use the top-k entities as hard negatives, excluding the correct one
    hard_negative_count = min(3, len(candidate_entities))  # Use up to 3 hard negatives
    hard_negatives = candidate_entities[:hard_negative_count]
    
    # Get entity IDs for hard negatives
    if 'entity_ids' in example and len(example['entity_ids']) > 1:
        # Find indices of hard negatives in topk_names
        hard_negative_indices = [example['topk_names'].index(hn) for hn in hard_negatives]
        # Get corresponding entity IDs
        hard_negative_ids = [example['entity_ids'][idx] for idx in hard_negative_indices]
        example['hard_negative_ids'] = hard_negative_ids
    
    # Update the example in-place   
    example['input'] = instruction
    return example


def load_kg_dataset(data_path: str) -> Dict[str, torch.utils.data.Dataset]:
    
    """Load and preprocess the knowledge graph dataset.
    
    Args:
        data_path: Path to the dataset
        
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    dataset = load_dataset(data_path) 
    dataset = dataset.shuffle().map(transform_kg_example)
     
    return dataset


def get_chatml_dataset(data_path: str):
    """Get a ChatML formatted dataset for knowledge graph completion.
    
    Args:
        data_path: Path to the dataset
        tokenizer: Tokenizer for encoding the text
        
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    return load_kg_dataset(data_path)


@dataclass
class KGDataCollator:
    """Collator for knowledge graph completion data with contrastive learning support.
    
    This collator prepares inputs for causal language modeling with KG special tokens
    and generates positive/negative pairs for contrastive learning.
    """
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    contrastive_learning: bool = True
    use_hard_negatives: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Process a batch of examples.
        
        Args:
            instances: Batch of examples from the dataset
            
        Returns:
            Dictionary with model inputs (input_ids, attention_mask, labels)
            and contrastive learning data if enabled
        """

        # Format sources and targets
        sources = [f"{self.tokenizer.bos_token} {example['input']}" for example in instances]
        targets = [f"{example['output']} {self.tokenizer.eos_token}" for example in instances]
        
        # Tokenize sources and targets
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        source_input_ids = tokenized_sources['input_ids']
        target_input_ids = tokenized_targets['input_ids']

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for source_ids, target_ids in zip(source_input_ids, target_input_ids):
            input_ids.append(torch.tensor(source_ids + target_ids))
            labels.append(
                torch.tensor([IGNORE_INDEX] * len(source_ids) + copy.deepcopy(target_ids))
            )

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # Create the batch dictionary
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels,
        }

        # Add KG-specific data
        data_dict['query_ids'] = torch.LongTensor([example['query_id'] for example in instances])
        data_dict['entity_ids'] = torch.LongTensor([example['entity_ids'] for example in instances])
        
        # Add contrastive learning data if enabled
        if self.contrastive_learning:
            # Create positive pair mask
            batch_size = len(instances)
            entity_count = data_dict['entity_ids'].size(1)
            
            # Initialize positive pairs mask
            # For each query, the correct entity is a positive pair
            positive_pairs = torch.zeros(batch_size, batch_size * entity_count, dtype=torch.bool)
            
            # For each query, mark its correct entity as a positive pair
            for i, example in enumerate(instances):
                # Get the index of the correct entity (usually the first one in entity_ids)
                correct_entity_idx = 0  # Assuming the first entity is the correct one
                
                # Mark the positive pair
                positive_pairs[i, i * entity_count + correct_entity_idx] = True
                
                # Also mark entities with the same output as positive pairs (if any)
                # This helps create more robust positive pairs
                correct_output = example['output']
                for j, other_example in enumerate(instances):
                    if i != j and other_example['output'] == correct_output:
                        # If another example has the same output, mark its entities as positive pairs
                        for k in range(entity_count):
                            if other_example['entity_ids'][k] == example['entity_ids'][correct_entity_idx]:
                                positive_pairs[i, j * entity_count + k] = True
            
            data_dict['positive_pairs'] = positive_pairs
            
            # Print debug info about dimensions
            print(f"Batch size: {batch_size}, Entity count: {entity_count}")
            print(f"Positive pairs shape: {positive_pairs.shape}")
            print(f"Positive pairs count: {positive_pairs.sum().item()}")
            
            # Add hard negative mining information if available and enabled
            if self.use_hard_negatives and all('hard_negative_ids' in example for example in instances):


                # Collect hard negative IDs from all instances
                hard_negative_ids_list = [example['hard_negative_ids'] for example in instances]
                
                # Determine max number of hard negatives per instance
                max_hard_negatives = max(len(hns) for hns in hard_negative_ids_list)
                
                # Pad hard negative lists to the same length
                padded_hard_negatives = []
                for hns in hard_negative_ids_list:
                    if len(hns) < max_hard_negatives:
                        # Pad with the last hard negative ID
                        padded = hns + [hns[-1] if hns else 0] * (max_hard_negatives - len(hns))
                    else:
                        padded = hns
                    padded_hard_negatives.append(padded)
                
                # Convert to tensor
                data_dict['hard_negative_ids'] = torch.LongTensor(padded_hard_negatives)
                
                # Create hard negative mask
                hard_negative_mask = torch.zeros(batch_size, batch_size * entity_count + max_hard_negatives, dtype=torch.bool)
                
                # Mark hard negatives in the mask
                for i in range(batch_size):
                    for j, hn_id in enumerate(padded_hard_negatives[i]):
                        hard_negative_mask[i, batch_size * entity_count + j] = True
                
                data_dict['hard_negative_mask'] = hard_negative_mask
                
                # Print debug info about hard negatives
                print(f"Hard negative IDs shape: {data_dict['hard_negative_ids'].shape}")
                print(f"Hard negative mask shape: {hard_negative_mask.shape}")
        
        return data_dict
