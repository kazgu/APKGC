#!/usr/bin/env python
# coding: utf-8
"""Utility functions for handling PyTorch tensors."""

import torch
import numpy as np


def convert_tensor_to_python(obj):
    """Convert PyTorch tensors to Python types for JSON serialization.
    
    Args:
        obj: Object that might contain tensors
        
    Returns:
        Object with tensors converted to Python types
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to Python type
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        # Recursively convert dictionaries
        return {k: convert_tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert lists
        return [convert_tensor_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        # Recursively convert tuples
        return tuple(convert_tensor_to_python(item) for item in obj)
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__
        return convert_tensor_to_python(obj.__dict__)
    else:
        # Return other types as is
        return obj
