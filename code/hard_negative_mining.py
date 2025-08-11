import json,random
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse
from transformers import AutoTokenizer
from model import EmbeddingModel

def load_dataset(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_dataset(data, file_path):
    """Save dataset to JSON file."""
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        else:
            return obj
    
    # Convert data to JSON serializable format
    serializable_data = convert_to_serializable(data)
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    print(f"Dataset saved to {file_path}")

def compute_entity_embeddings(embedding_dir, device='cuda'):
    """Load entity embeddings and compute their similarities."""
    # Load entity embeddings
    entity_embedding_path = os.path.join(embedding_dir, 'entity_embeddings.pt')
    entity_embeddings = torch.load(entity_embedding_path, map_location=device)
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = torch.nn.functional.normalize(entity_embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarities
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    return entity_embeddings, similarity_matrix

def find_hard_negatives_by_similarity(dataset, similarity_matrix, k=10):
    """Find hard negatives based on embedding similarity."""
    print("Finding hard negatives based on embedding similarity...")
    
    for example in tqdm(dataset):
        # Get the correct entity ID
        correct_entity_idx = example['entity_ids'][0]  # Assuming the first entity is the correct one
        
        # Get candidate entity IDs (excluding the correct one)
        candidate_entity_ids = example['entity_ids'][1:]
        
        # Get similarities between the correct entity and all entities
        entity_similarities = similarity_matrix[correct_entity_idx].cpu().numpy()
        
        # Sort entities by similarity to the correct entity (excluding the correct entity itself)
        # We set the similarity of the correct entity to -1 to exclude it
        entity_similarities[correct_entity_idx] = -1
        
        # Get the most similar entities (hard negatives)
        # Increase k to get more hard negatives
        hard_negative_indices = np.argsort(entity_similarities)[::-1][:k]
        
        # Filter out hard negatives that are too similar (potential false negatives)
        # This helps avoid confusing the model with entities that might actually be correct
        similarity_threshold = 0.95  # Adjust based on your embedding space
        filtered_hard_negatives = []
        for idx in hard_negative_indices:
            sim = similarity_matrix[correct_entity_idx, idx].item()
            if sim < similarity_threshold:
                filtered_hard_negatives.append(idx)
            if len(filtered_hard_negatives) >= k:
                break
        
        # If we don't have enough hard negatives after filtering, add some random ones
        if len(filtered_hard_negatives) < k:
            all_indices = set(range(len(similarity_matrix)))
            all_indices.remove(correct_entity_idx)
            all_indices = all_indices - set(filtered_hard_negatives)
            random_indices = random.sample(list(all_indices), min(k - len(filtered_hard_negatives), len(all_indices)))
            filtered_hard_negatives.extend(random_indices)
        
        # Store hard negative IDs
        example['hard_negative_ids'] = filtered_hard_negatives[:k]
        
        # Print some statistics about the selected hard negatives
        if len(example['hard_negative_ids']) > 0:
            hard_neg_sims = [similarity_matrix[correct_entity_idx, idx].item() for idx in example['hard_negative_ids']]
            avg_sim = sum(hard_neg_sims) / len(hard_neg_sims)
            max_sim = max(hard_neg_sims)
            min_sim = min(hard_neg_sims)
            print(f"Hard negative stats - Avg: {avg_sim:.4f}, Max: {max_sim:.4f}, Min: {min_sim:.4f}")
    
    return dataset

def find_hard_negatives_by_relation(dataset, k=5):
    """Find hard negatives based on relation patterns."""
    print("Finding hard negatives based on relation patterns...")
    
    # Group examples by relation
    relation_examples = defaultdict(list)
    for i, example in enumerate(dataset):
        relation = example['triplet'][1]  # Relation is the second element in the triplet
        relation_examples[relation].append(i)
    
    # For each example, find hard negatives from the same relation
    for example_idx, example in enumerate(tqdm(dataset)):
        relation = example['triplet'][1]
        correct_entity = example['output']
        
        # Get other examples with the same relation
        related_examples = [dataset[i] for i in relation_examples[relation] if i != example_idx]
        
        # Get entities from related examples (excluding the correct entity)
        related_entities = []
        for rel_ex in related_examples:
            if rel_ex['output'] != correct_entity:
                related_entities.append({
                    'name': rel_ex['output'],
                    'id': rel_ex['entity_ids'][0]  # Assuming the first entity is the correct one
                })
        
        # Select up to k hard negatives
        hard_negatives = related_entities[:k]
        
        # Store hard negative IDs
        example['hard_negative_ids_by_relation'] = [hn['id'] for hn in hard_negatives]
        example['hard_negative_names_by_relation'] = [hn['name'] for hn in hard_negatives]
    
    return dataset

def find_hard_negatives_by_context(dataset, k=5):
    """Find hard negatives based on context similarity."""
    print("Finding hard negatives based on context similarity...")
    
    # Extract context from each example
    contexts = []
    for example in dataset:
        # Extract context from the input field
        input_text = example['input']
        # Use a more robust way to extract context
        try:
            # Try to find context in the input
            if "entity_descriptions" in input_text:
                # Extract from the entity descriptions section
                start_idx = input_text.find("entity_descriptions")
                end_idx = input_text.find("supporting_triples")
                if start_idx != -1 and end_idx != -1:
                    context = input_text[start_idx:end_idx]
                else:
                    context = ""
            else:
                # Fallback to a simpler approach
                context = input_text[:min(500, len(input_text))]  # Use first 500 chars as context
        except Exception as e:
            print(f"Error extracting context: {e}")
            context = ""
        
        contexts.append(context)
    
    # For simplicity, we'll use a basic string matching approach
    # In a real implementation, you might use embeddings or more sophisticated NLP
    for i, example in enumerate(tqdm(dataset)):
        try:
            context_i = contexts[i] if i < len(contexts) else ""
            correct_entity = example['output']
            
            # Compute similarity with other contexts
            similarities = []
            for j, other_example in enumerate(dataset):
                if i != j and j < len(contexts) and other_example['output'] != correct_entity:
                    try:
                        # Simple similarity: count of common words
                        # Handle empty contexts gracefully
                        if not context_i or not contexts[j]:
                            similarity = 0
                        else:
                            # Split with error handling
                            words_i = context_i.lower().split() if context_i else []
                            words_j = contexts[j].lower().split() if contexts[j] else []
                            common_words = set(words_i) & set(words_j)
                            similarity = len(common_words)
                        
                        similarities.append((similarity, j))
                    except Exception as e:
                        print(f"Error computing similarity between examples {i} and {j}: {e}")
            
            # Sort by similarity (descending)
            similarities.sort(reverse=True)
            
            # Select top-k hard negatives
            hard_negative_indices = []
            hard_negative_names = []
            
            for _, j in similarities[:k]:
                try:
                    if 'entity_ids' in dataset[j] and len(dataset[j]['entity_ids']) > 0:
                        hard_negative_indices.append(dataset[j]['entity_ids'][0])
                        hard_negative_names.append(dataset[j]['output'])
                except Exception as e:
                    print(f"Error adding hard negative from example {j}: {e}")
            
            # Ensure we have exactly k hard negatives
            while len(hard_negative_indices) < k:
                # Add a random entity ID as a fallback
                random_idx = np.random.randint(0, len(dataset))
                if random_idx != i and 'entity_ids' in dataset[random_idx] and len(dataset[random_idx]['entity_ids']) > 0:
                    hard_negative_indices.append(dataset[random_idx]['entity_ids'][0])
                    hard_negative_names.append(dataset[random_idx]['output'])
            
            # Store hard negative IDs and names
            example['hard_negative_ids_by_context'] = hard_negative_indices[:k]
            example['hard_negative_names_by_context'] = hard_negative_names[:k]
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            # Provide default values
            example['hard_negative_ids_by_context'] = [0] * k
            example['hard_negative_names_by_context'] = [""] * k
    
    return dataset

def combine_hard_negatives(dataset, methods=['similarity', 'relation', 'context'], k_per_method=5, k_total=10):
    """Combine hard negatives from different methods."""
    print("Combining hard negatives from different methods...")
    
    for example in tqdm(dataset):
        all_hard_negatives = []
        
        # Collect hard negatives from each method
        if 'similarity' in methods and 'hard_negative_ids' in example:
            all_hard_negatives.extend([(id, 'similarity') for id in example['hard_negative_ids'][:k_per_method]])
        
        if 'relation' in methods and 'hard_negative_ids_by_relation' in example:
            all_hard_negatives.extend([(id, 'relation') for id in example['hard_negative_ids_by_relation'][:k_per_method]])
        
        if 'context' in methods and 'hard_negative_ids_by_context' in example:
            all_hard_negatives.extend([(id, 'context') for id in example['hard_negative_ids_by_context'][:k_per_method]])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_hard_negatives = []
        for id, method in all_hard_negatives:
            if id not in seen:
                seen.add(id)
                unique_hard_negatives.append((id, method))
        
        # If we don't have enough hard negatives, add some random ones
        if len(unique_hard_negatives) < k_total:
            # Get the correct entity ID
            correct_entity_idx = example['entity_ids'][0]
            
            # Get all entity IDs
            all_entity_ids = set(range(len(example['entity_ids'])))
            
            # Remove the correct entity and already selected hard negatives
            all_entity_ids.remove(correct_entity_idx)
            for id, _ in unique_hard_negatives:
                if id in all_entity_ids:
                    all_entity_ids.remove(id)
            
            # Add random entities as hard negatives
            random_entities = random.sample(list(all_entity_ids), min(k_total - len(unique_hard_negatives), len(all_entity_ids)))
            unique_hard_negatives.extend([(id, 'random') for id in random_entities])
        
        # Select top-k hard negatives
        final_hard_negatives = unique_hard_negatives[:k_total]
        
        # Store final hard negative IDs
        example['hard_negative_ids'] = [id for id, _ in final_hard_negatives]
        example['hard_negative_methods'] = [method for _, method in final_hard_negatives]
        
        # Print statistics about the hard negatives
        method_counts = {}
        for _, method in final_hard_negatives:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print(f"Hard negative methods: {method_counts}")
    
    return dataset

def main(args):
    # Load dataset
    dataset = load_dataset(args.input_file)
    print(f"Loaded {len(dataset)} examples from {args.input_file}")
    
    # Compute entity embeddings and similarities
    entity_embeddings, similarity_matrix = compute_entity_embeddings(args.embedding_dir)
    
    # Find hard negatives using different methods
    if 'similarity' in args.methods:
        dataset = find_hard_negatives_by_similarity(dataset, similarity_matrix, k=args.k_per_method)
    
    if 'relation' in args.methods:
        dataset = find_hard_negatives_by_relation(dataset, k=args.k_per_method)
    
    if 'context' in args.methods:
        dataset = find_hard_negatives_by_context(dataset, k=args.k_per_method)
    
    # Combine hard negatives from different methods
    dataset = combine_hard_negatives(dataset, methods=args.methods, k_per_method=args.k_per_method, k_total=args.k_total)
    
    # Save updated dataset
    save_dataset(dataset, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hard negative mining for knowledge graph completion")
    parser.add_argument("--input_file", type=str, required=True, help="Input dataset file")
    parser.add_argument("--output_file", type=str, required=True, help="Output dataset file with hard negatives")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Directory containing entity embeddings")
    parser.add_argument("--methods", type=str, nargs="+", default=["similarity", "relation", "context"], 
                        help="Hard negative mining methods to use")
    parser.add_argument("--k_per_method", type=int, default=3, help="Number of hard negatives per method")
    parser.add_argument("--k_total", type=int, default=5, help="Total number of hard negatives to keep")
    
    args = parser.parse_args()
    main(args)
