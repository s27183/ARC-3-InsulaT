"""
Decision Transformer Agent Module

This module provides Pure Decision Transformer implementations for ARC-AGI-3.
The Pure DT replaces the hybrid bandit-DT architecture with a unified transformer
approach for direct action prediction.
"""

from .config import (
    load_dt_config,
    get_loss_config_summary,
    validate_config,
    DEFAULT_DT_CONFIG,
    CROSS_ENTROPY_CONFIG,
    SELECTIVE_CONFIG,
    HYBRID_CONFIG
)

from .dt_agent import DTAgent

__all__ = [
    # Configuration
        'load_dt_config',
    'get_loss_config_summary', 
    'validate_config',
        'DEFAULT_DT_CONFIG',
    'CROSS_ENTROPY_CONFIG',
    'SELECTIVE_CONFIG', 
    'HYBRID_CONFIG',
    # Model,
        "DTAgent"
]