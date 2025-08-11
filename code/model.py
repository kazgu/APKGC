"""Knowledge Graph Embedding-based Language Model for KG completion."""

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.activations import ACT2FN
from peft import PeftModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism implementation."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention layer.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for query, key, value
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len, hidden_dim]
            key: Key tensor of shape [batch_size, seq_len, hidden_dim]
            value: Value tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        out = torch.matmul(attn, v)
        
        # Reshape and project to output dimension
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class AttentionAdapter(nn.Module):
    """Attention-based adapter for embedding transformation."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize attention adapter.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input embeddings using attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Input projection
        x = self.input_projection(x)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Self-attention block with residual connection
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward block with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection
        out = self.output_projection(x)  # [batch_size, output_dim]
        
        return out


class EmbeddingModel(nn.Module):
    """Model for transforming KG embeddings with contrastive learning support."""
    
    def __init__(
            self, 
            embedding_dir: str, 
            input_size: int, 
            intermediate_size: int = 1024,
            output_size: int = 4096, 
            hidden_act: str = 'silu',
            num_heads: int = 4,
            dropout: float = 0.1,
            contrastive_dim: int = 256
    ) -> None:
        """Initialize embedding model.
        
        Args:
            embedding_dir: Directory containing embeddings
            input_size: Size of input embeddings
            intermediate_size: Size of intermediate representations
            output_size: Size of output embeddings
            hidden_act: Activation function for hidden layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Load pre-trained embeddings
        entity_embedding_path = os.path.join(embedding_dir, 'entity_embeddings.pt')
        query_embedding_path = os.path.join(embedding_dir, 'query_embeddings.pt')

        entity_embeddings = torch.load(entity_embedding_path)
        # Allow embeddings to be fine-tuned
        entity_embeddings.requires_grad = True
        self.ent_embeddings = nn.Embedding.from_pretrained(entity_embeddings).cuda()

        query_embeddings = torch.load(query_embedding_path)
        # Allow embeddings to be fine-tuned
        query_embeddings.requires_grad = True
        self.query_embeddings = nn.Embedding.from_pretrained(query_embeddings).cuda()
        
        # Initialize adapter with attention mechanism
        self.adapter = nn.Sequential(
            AttentionAdapter(
                input_dim=input_size,
                hidden_dim=intermediate_size,
                output_dim=output_size,
                num_heads=num_heads,
                dropout=dropout
            )
        ).cuda()
        
        # Contrastive learning projection head
        self.contrastive_projection = nn.Sequential(
            nn.Linear(output_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, contrastive_dim)
        ).cuda()
        
        # Fixed temperature parameter for contrastive loss (non-learnable)
        self.temperature = torch.tensor(0.07).cuda()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for layer in self.adapter:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, query_ids: torch.Tensor, entity_ids: torch.Tensor, 
                return_projections: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Transform query and entity embeddings.
        
        Args:
            query_ids: Tensor of query IDs of shape [batch_size]
            entity_ids: Tensor of entity IDs of shape [batch_size * K]
            return_projections: Whether to return contrastive projections
            
        Returns:
            If return_projections is False:
                Tuple of (query_embeds, entity_embeds) where:
                    query_embeds: Transformed query embeddings [batch_size, hidden_size]
                    entity_embeds: Transformed entity embeddings [batch_size * K, hidden_size]
            If return_projections is True:
                Tuple of (query_embeds, entity_embeds, query_proj, entity_proj) where:
                    query_proj: Contrastive projections of queries [batch_size, contrastive_dim]
                    entity_proj: Contrastive projections of entities [batch_size * K, contrastive_dim]
        """
        # Get embeddings
        query_embeds = self.query_embeddings(query_ids)
        entity_embeds = self.ent_embeddings(entity_ids)
        
        # Transform embeddings through adapter
        query_embeds = self.adapter(query_embeds)
        entity_embeds = self.adapter(entity_embeds)
        
        if not return_projections:
            return query_embeds, entity_embeds
        
        # Project embeddings for contrastive learning
        query_proj = F.normalize(self.contrastive_projection(query_embeds), p=2, dim=1)
        entity_proj = F.normalize(self.contrastive_projection(entity_embeds), p=2, dim=1)
        
        return query_embeds, entity_embeds, query_proj, entity_proj
    
    def compute_contrastive_loss(self, query_proj: torch.Tensor, entity_proj: torch.Tensor, 
                                positive_pairs: torch.Tensor,
                                hard_negative_ids: Optional[torch.Tensor] = None,
                                hard_negative_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute contrastive loss between queries and entities with hard negative mining.
        
        Args:
            query_proj: Contrastive projections of queries [batch_size, contrastive_dim]
            entity_proj: Contrastive projections of entities [batch_size * K, contrastive_dim]
            positive_pairs: Boolean mask indicating positive pairs [batch_size, batch_size * K]
            hard_negative_ids: IDs of hard negative entities [batch_size, H]
            hard_negative_mask: Boolean mask for hard negatives [batch_size, batch_size * K + H]
            
        Returns:
            Contrastive loss value
        """
        batch_size = query_proj.size(0)
        
        # Double-check normalization to ensure cosine similarity is properly bounded
        query_proj = F.normalize(query_proj, p=2, dim=1)
        entity_proj = F.normalize(entity_proj, p=2, dim=1)
        
        # Use a higher temperature for more stable gradients
        effective_temperature = 0.1
        
        # Compute similarity matrix between queries and entities with effective temperature
        similarity = torch.matmul(query_proj, entity_proj.transpose(0, 1)) / effective_temperature
        
        # Apply stronger label smoothing to positive pairs to avoid overconfidence
        # This helps prevent the model from becoming too confident in its predictions
        label_smoothing = 0.2
        smoothed_positive_pairs = positive_pairs.float() * (1 - label_smoothing) + label_smoothing / positive_pairs.sum(dim=1, keepdim=True).clamp(min=1).float()
        
        # If hard negatives are provided, incorporate them into the loss calculation
        if hard_negative_ids is not None and hard_negative_mask is not None:
            try:
                # Get embeddings for hard negatives
                hard_negative_embeds = self.ent_embeddings(hard_negative_ids.view(-1))
                hard_negative_embeds = self.adapter(hard_negative_embeds)
                hard_negative_proj = F.normalize(self.contrastive_projection(hard_negative_embeds), p=2, dim=1)
                
                # Reshape to [batch_size, H, contrastive_dim]
                num_hard_negatives = hard_negative_ids.size(1) if hard_negative_ids.dim() > 1 else 1
                hard_negative_proj = hard_negative_proj.view(batch_size, num_hard_negatives, query_proj.size(1))
                
                # Compute similarity between queries and hard negatives
                hard_negative_sim = torch.bmm(
                    query_proj.unsqueeze(1),  # [batch_size, 1, contrastive_dim]
                    hard_negative_proj.transpose(1, 2)  # [batch_size, contrastive_dim, H]
                ).squeeze(1) / effective_temperature  # [batch_size, H]
                
                # Combine regular similarities with hard negative similarities
                combined_sim = torch.cat([similarity, hard_negative_sim], dim=1)
                
                # Use NT-Xent loss (normalized temperature-scaled cross entropy)
                # This is a more stable version of the InfoNCE loss
                
                # Create mask for positive pairs
                extended_positive_pairs = torch.zeros_like(combined_sim, dtype=torch.float, device=positive_pairs.device)
                extended_positive_pairs[:, :positive_pairs.size(1)] = smoothed_positive_pairs
                
                # Create mask for all pairs (including hard negatives)
                all_pairs_mask = torch.ones_like(combined_sim, dtype=torch.float, device=positive_pairs.device)
                
                # Compute NT-Xent loss
                exp_sim = torch.exp(combined_sim)
                
                # Sum of exp similarities for positive pairs
                pos_sim = torch.sum(exp_sim * extended_positive_pairs, dim=1)
                
                # Sum of exp similarities for all pairs
                all_sim = torch.sum(exp_sim * all_pairs_mask, dim=1)
                
                # Compute loss: -log(pos_sim / all_sim)
                # Add a small epsilon to avoid numerical instability
                epsilon = 1e-8
                contrastive_loss = -torch.log((pos_sim + epsilon) / (all_sim + epsilon))
                
                # Add a direct supervised loss component to explicitly push hard negatives away
                # This helps when the contrastive loss is not decreasing
                supervised_loss = torch.tensor(0.0, device=contrastive_loss.device)
                
                # For each query, get the similarity with its positive entity
                for i in range(batch_size):
                    # Find the indices of positive pairs for this query
                    pos_indices = positive_pairs[i].nonzero().view(-1)
                    if len(pos_indices) > 0:
                        # Get the average similarity with positive entities
                        pos_sim_i = similarity[i, pos_indices].mean()
                        
                        # Get the similarity with hard negatives
                        if hard_negative_sim.size(1) > 0:
                            # Calculate how much the hard negatives are closer to the query than they should be
                            # We want positive similarity to be higher than hard negative similarity by at least the margin
                            margin = 0.5  # Larger margin to create more separation
                            hard_neg_violations = torch.relu(hard_negative_sim[i] - pos_sim_i + margin)
                            supervised_loss = supervised_loss + hard_neg_violations.mean()
                
                # Normalize supervised loss by batch size
                supervised_loss = supervised_loss / batch_size
                
                # Combine the losses with a higher weight on supervised loss
                contrastive_loss = contrastive_loss.mean() + 1.0 * supervised_loss
                
            except Exception as e:
                print(f"Error in contrastive loss with hard negatives: {e}")
                print(f"Falling back to standard contrastive loss")
                # Fall back to standard contrastive loss with improved numerical stability
                exp_sim = torch.exp(similarity)
                pos_sim = torch.sum(exp_sim * smoothed_positive_pairs, dim=1)
                all_sim = torch.sum(exp_sim, dim=1)
                epsilon = 1e-8
                contrastive_loss = -torch.log((pos_sim + epsilon) / (all_sim + epsilon)).mean()
        else:
            # Standard contrastive loss without hard negatives, with improved numerical stability
            exp_sim = torch.exp(similarity)
            pos_sim = torch.sum(exp_sim * smoothed_positive_pairs, dim=1)
            all_sim = torch.sum(exp_sim, dim=1)
            epsilon = 1e-8
            contrastive_loss = -torch.log((pos_sim + epsilon) / (all_sim + epsilon)).mean()
        
        return contrastive_loss


class KGELLM(nn.Module):
    """Knowledge Graph Embedding enhanced Language Model."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        llama_model: Union[LlamaForCausalLM, PeftModel], 
        kge_model: EmbeddingModel,
    ):
        """Initialize KGELLM model.
        
        Args:
            tokenizer: Tokenizer for encoding text
            llama_model: Base language model
            kge_model: Knowledge graph embedding model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.llama_model = llama_model
        self.kge_model = kge_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        query_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        **kwargs  # To handle any additional unused arguments
    ):
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            query_ids: Query IDs for KG embeddings
            entity_ids: Entity IDs for KG embeddings
            
        Returns:
            Output from the language model
        """
        # Get special token IDs
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        
        # Find positions of special tokens
        query_position = torch.nonzero(input_ids == query_holder)  # (batch_size, 2)
        entity_position = torch.nonzero(input_ids == entity_holder) # (batch_size*K, 2)

        # Get KG embeddings
        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1)) 

        # Replace special tokens with pad tokens
        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == query_holder] = self.tokenizer.pad_token_id
        input_ids_copy[input_ids_copy == entity_holder] = self.tokenizer.pad_token_id
        
        # Get token embeddings
        inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids_copy).clone()

        # Match dtype of model embeddings
        query_embeds = query_embeds.to(dtype=inputs_embeds.dtype)
        entity_embeds = entity_embeds.to(dtype=inputs_embeds.dtype)

        # Insert KG embeddings at special token positions
        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds[:entity_position.shape[0]]

        # Forward pass through the language model
        return self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    def save_pretrained(self, peft_model_path: str) -> None:
        """Save model weights.
        
        Args:
            peft_model_path: Path to save the model
        """
        print('Saving model to:', peft_model_path)
        self.llama_model.save_pretrained(peft_model_path)
        torch.save(self.kge_model.state_dict(), os.path.join(peft_model_path, 'kge.bin'))

    def generate(
        self,
        input_ids: torch.LongTensor,
        query_ids: torch.LongTensor,
        entity_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        use_cache: bool = True,
        **generation_kwargs
    ) -> torch.LongTensor:
        """Generate text with KG embeddings.
        
        Args:
            input_ids: Input token IDs
            query_ids: Query IDs for KG embeddings
            entity_ids: Entity IDs for KG embeddings
            max_new_tokens: Maximum number of new tokens to generate
            use_cache: Whether to use KV cache during generation
            generation_kwargs: Additional keyword arguments for generation
            
        Returns:
            Generated token IDs
        """
        # Get special token IDs
        query_holder = self.tokenizer.convert_tokens_to_ids(['[QUERY]'])[0]
        entity_holder = self.tokenizer.convert_tokens_to_ids(['[ENTITY]'])[0]
        
        # Find positions of special tokens
        query_position = torch.nonzero(input_ids == query_holder)  # (batch_size, 2)
        entity_position = torch.nonzero(input_ids == entity_holder) # (batch_size*K, 2)

        # Get KG embeddings
        query_embeds, entity_embeds = self.kge_model(query_ids, entity_ids.view(-1)) 

        # Replace special tokens with pad tokens
        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == query_holder] = self.tokenizer.pad_token_id
        input_ids_copy[input_ids_copy == entity_holder] = self.tokenizer.pad_token_id
        
        # Get token embeddings
        inputs_embeds = self.llama_model.model.model.embed_tokens(input_ids_copy).clone()

        # Match dtype of model embeddings
        query_embeds = query_embeds.to(dtype=inputs_embeds.dtype)
        entity_embeds = entity_embeds.to(dtype=inputs_embeds.dtype)

        # Ensure embeddings are on the correct device
        query_embeds = query_embeds.cuda()
        query_position = query_position.cuda()
        entity_position = entity_position.cuda()
        entity_embeds = entity_embeds.cuda()
        
        # Insert KG embeddings at special token positions
        inputs_embeds[query_position[:, 0], query_position[:, 1]] = query_embeds
        inputs_embeds[entity_position[:, 0], entity_position[:, 1]] = entity_embeds[:entity_position.shape[0]]

        # Generate text with the language model
        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            **generation_kwargs
        )
