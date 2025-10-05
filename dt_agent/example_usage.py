#!/usr/bin/env python3
"""
Example usage of the self-contained Pure Decision Transformer Agent.

This example demonstrates how to use the PureDTAgent directly without
going through the custom_agents/action.py module.
"""

from dt_agent import PureDTAgent, load_pure_dt_config


def run_pure_dt_agent_example():
    """Example of how to instantiate and configure a Pure DT Agent."""
    
    # Example agent parameters (normally provided by ARC-AGI-3 framework)
    agent_params = {
        'card_id': 'example_card_123',
        'game_id': 'example_game_456', 
        'agent_name': 'PureDTAgent_Example',
        'ROOT_URL': 'https://arc-agi-3.com',  # Example URL
        'record': False,  # Set to True to enable recording
        'tags': ['pure_dt', 'transformer', 'example']
    }
    
    try:
        # Initialize the Pure DT Agent - fully self-contained
        print("🤖 Initializing Pure Decision Transformer Agent...")
        agent = PureDTAgent(**agent_params)
        
        # The agent is now ready to use with the ARC-AGI-3 framework
        print(f"✅ Pure DT Agent initialized successfully!")
        print(f"   - Device: {agent.device}")
        print(f"   - Model: {agent.pure_dt_model.__class__.__name__}")
        print(f"   - Configuration: {agent.pure_dt_config['loss_type']} loss")
        print(f"   - Context length: {agent.pure_dt_config['context_length']}")
        print(f"   - Temperature: {agent.pure_dt_config['temperature']}")
        
        # Example configuration customization
        print("\n🔧 Configuration options:")
        config = load_pure_dt_config()
        print(f"   - Available loss types: cross_entropy, selective, hybrid")
        print(f"   - Current loss type: {config['loss_type']}")
        print(f"   - Embed dimension: {config['embed_dim']}")
        print(f"   - Number of layers: {config['num_layers']}")
        print(f"   - Training frequency: every {config['train_frequency']} actions")
        
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
        ('cross_entropy', "Dense updates on all actions"),
        ('selective', "Sparse updates only on positive rewards"),
        ('hybrid', "Confidence-based interpolation between dense/sparse")
    ]
    
    for loss_type, description in configs:
        print(f"• {loss_type:15} - {description}")
    
    print(f"\n💡 To change configuration, modify dt_agent/pure_dt_config.py")
    print(f"   or create custom config files for different experiments.")


def main():
    """Main example function."""
    print("=" * 60)
    print("Pure Decision Transformer Agent - Standalone Example")
    print("=" * 60)
    
    # Run the main example
    agent = run_pure_dt_agent_example()
    
    # Show configuration options
    compare_configurations()
    
    print(f"\n🎯 Key Benefits of Self-Contained Pure DT Agent:")
    print(f"   ✓ No dependency on custom_agents/action.py")
    print(f"   ✓ Direct import: from dt_agent import PureDTAgent")
    print(f"   ✓ Full Agent interface compatibility") 
    print(f"   ✓ All infrastructure included (training, logging, viz)")
    print(f"   ✓ Configurable loss functions for different strategies")
    
    if agent:
        print(f"\n✅ Ready to replace custom_agents.Action with dt_agent.PureDTAgent!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()