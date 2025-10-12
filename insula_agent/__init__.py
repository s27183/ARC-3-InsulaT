"""
Insula Module

This module provides Insula - an online supervised learning architecture
with insular cortex-inspired multi-level integration for ARC-AGI-3.
"""

from insula_agent.config import (
    load_dt_config,
    get_loss_config_summary,
    validate_config,
    DEFAULT_DT_CONFIG,
)

from insula_agent.insula import Insula

__all__ = [
    # Configuration
    "load_dt_config",
    "get_loss_config_summary",
    "validate_config",
    "DEFAULT_DT_CONFIG",
    # Model
        "Insula",
]
