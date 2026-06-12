"""
Base model class for disease severity prediction models.

This module provides an abstract base class that all model implementations
should inherit from to ensure consistency and proper integration with the
training pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models.
    
    This class enforces a consistent interface for all model implementations
    and provides common utilities for model management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config or {}
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor for regression task of shape (B, 1) or (B,)
        """
        pass
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters (implementation-specific).
        Should be overridden by subclasses that have a backbone.
        """
        raise NotImplementedError(
            f"{self.model_name} does not implement freeze_backbone()"
        )
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters (implementation-specific).
        Should be overridden by subclasses that have a backbone.
        """
        raise NotImplementedError(
            f"{self.model_name} does not implement unfreeze_backbone()"
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_name': self.model_name,
            'num_parameters': self.get_num_parameters(),
            'num_trainable_parameters': self.get_num_parameters(trainable_only=True),
            **self.config
        }
    
    def summary(self) -> str:
        """
        Get a summary string of the model.
        
        Returns:
            String containing model summary information
        """
        total_params = self.get_num_parameters()
        trainable_params = self.get_num_parameters(trainable_only=True)
        
        summary_str = f"""
{'='*60}
Model: {self.model_name}
{'='*60}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}
{'='*60}
        """
        return summary_str.strip()
