#!/usr/bin/env python3
"""
Example usage of the self-contained Pure Decision Transformer Agent.

This example demonstrates how to use the PureDTAgent directly without
going through the custom_agents/action.py module.
"""

from insula_agent import Insula, load_config


def run_pure_dt_agent_example():
    """Example of how to instantiate and configure an Insula Agent.

    NOTE: This example is outdated and needs to be rewritten for the new Insula architecture.
    """

    # Example agent parameters (normally provided by ARC-AGI-3 framework)
    agent_params = {
        "card_id": "example_card_123",
        "game_id": "example_game_456",
        "agent_name": "InsulaAgent_Example",
        "ROOT_URL": "https://arc-agi-3.com",  # Example URL
        "record": False,  # Set to True to enable recording
        "tags": ["insula", "transformer", "example"],
    }

    try:
        # Initialize the Insula Agent - fully self-contained
        print("🤖 Initializing Insula Agent...")
        agent = Insula(**agent_params)

        # The agent is now ready to use with the ARC-AGI-3 framework
        print(f"✅ Insula Agent initialized successfully!")
        print(f"   - Device: {agent.device}")
        print(f"   - Model: {agent.decision_model.__class__.__name__}")
        print(f"   - Configuration: {agent.config.summary()}")

        # Example configuration customization
        print("\n🔧 Configuration options:")
        config = load_config()
        print(f"   - Embed dimension: {config.embed_dim}")
        print(f"   - Number of layers: {config.num_layers}")
        print(f"   - Training frequency: every {config.train_frequency} actions")

        # The agent can now be used with agent.main() in the ARC-AGI-3 framework
        print(f"\n🚀 Agent ready! Use agent.main() to start the game loop.")

        return agent

    except Exception as e:
        print(f"❌ Error initializing Pure DT Agent: {e}")
        return None


def compare_configurations():
    """Example of different Pure DT configurations."""

    print("\n📊 Pure DT Configuration Comparison:")
    print("-" * 50)

    configs = [
        ("cross_entropy", "Dense updates on all actions"),
        ("selective", "Sparse updates only on positive rewards"),
        ("hybrid", "Confidence-based interpolation between dense/sparse"),
    ]

    for loss_type, description in configs:
        print(f"• {loss_type:15} - {description}")

    print(f"\n💡 To change configuration, modify dt_agent/pure_dt_config.py")
    print(f"   or create custom config files for different experiments.")


def demonstrate_vit_configuration():
    """Demonstrate ViT state encoder configuration options."""

    print("\n🔬 Vision Transformer State Encoder Configuration:")
    print("-" * 50)

    # Load default config to show ViT settings
    config = load_config()

    print(f"\n  Current ViT Settings:")
    print(f"    • Cell embedding dim: {config.vit_cell_embed_dim}")
    print(
        f"      → Each color (0-15) maps to {config.vit_cell_embed_dim}-dim learned vector"
    )
    print(
        f"    • Patch size: {config.vit_patch_size}×{config.vit_patch_size}"
    )
    print(
        f"      → Creates {(64 // config.vit_patch_size) ** 2} patches from 64×64 grid"
    )
    print(
        f"      → Each patch: {config.vit_patch_size ** 2} cells (not {16 * config.vit_patch_size ** 2} one-hot values)"
    )
    print(f"    • ViT layers: {config.vit_num_layers}")
    print(f"    • ViT attention heads: {config.vit_num_heads}")
    print(f"    • Dropout: {config.vit_dropout}")
    print(f"    • Use CLS token: {config.vit_use_cls_token}")

    print(f"\n  ViT Architecture Benefits:")
    print(f"    ✓ Learned cell embeddings (like word embeddings in NLP)")
    print(f"    ✓ 16× more efficient than one-hot encoding per patch")
    print(f"    ✓ 8× smaller input tensors (64×64 integers vs 16×64×64 floats)")
    print(f"    ✓ Global attention from layer 1 (non-local causality)")
    print(f"    ✓ Pure transformer hierarchy (ViT spatial + Transformer temporal)")
    print(f"    ✓ Learned spatial relationships via self-attention")
    print(f"    ✓ Efficient: O(n²) where n=64 patches (not 4096 cells)")

    print(f"\n  Patch Size Trade-offs:")
    print(f"    • 4×4 patches → 256 patches, 16 cells each (fine-grained, slower)")
    print(f"    • 8×8 patches → 64 patches, 64 cells each (balanced, recommended)")
    print(f"    • 16×16 patches → 16 patches, 256 cells each (coarse, faster)")

    print(f"\n  💡 To customize ViT, create a custom InsulaConfig:")
    print(f"     from insula_agent import InsulaConfig")
    print(f"     config = InsulaConfig(")
    print(f"         vit_cell_embed_dim=64,")
    print(f"         vit_patch_size=8,")
    print(f"         vit_num_layers=4,")
    print(f"         vit_num_heads=8,")
    print(f"     )")


def main():
    """Main example function."""
    print("=" * 60)
    print("Pure Decision Transformer Agent - Standalone Example")
    print("=" * 60)

    # Run the main example
    agent = run_pure_dt_agent_example()

    # Show configuration options
    compare_configurations()

    # Demonstrate ViT configuration
    demonstrate_vit_configuration()

    print(f"\n🎯 Key Benefits of Self-Contained Pure DT Agent:")
    print(f"   ✓ No dependency on custom_agents/action.py")
    print(f"   ✓ Direct import: from dt_agent import DTAgent")
    print(f"   ✓ Full Agent interface compatibility")
    print(f"   ✓ All infrastructure included (training, logging, viz)")
    print(f"   ✓ Configurable loss functions for different strategies")
    print(f"   ✓ Vision Transformer for global spatial reasoning")

    if agent:
        print(f"\n✅ Ready to replace custom_agents.Action with dt_agent.DTAgent!")

    print("=" * 60)


if __name__ == "__main__":
    main()
