"""
Insula Module

This module provides Insula - an online supervised learning architecture
with insular cortex-inspired multi-level integration for ARC-AGI-3.
"""

from insula_agent.config import (
    InsulaConfig,
    load_config,
    default_config,
    cpu_config,
    gpu_config,
)

from insula_agent.insula import Insula

__all__ = [
    # Configuration
    "InsulaConfig",
    "load_config",
    "default_config",
    "cpu_config",
    "gpu_config",
    # Model
    "Insula",
]
