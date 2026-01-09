"""
Test script to verify classification_agent configuration in agents.yaml
and that the prompt file exists and is accessible.
"""

import yaml
from pathlib import Path


def test_classification_agent_config():
    """Test that classification_agent is properly configured in agents.yaml."""
    # Get the path to agents.yaml (same logic as BaseAgent)
    # Assuming we're running from project root or tests directory
    project_root = Path(__file__).parent.parent
    orchestrator_dir = project_root / "apps" / "orchestrator"
    settings_dir = orchestrator_dir / "settings"
    config_path = settings_dir / "agents.yaml"
    
    print(f"Loading agents.yaml from: {config_path}")
    print(f"Config file exists: {config_path.exists()}")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print(f"\nAvailable agents: {list(config.keys())}")
    
    # Verify classification_agent exists
    if "classification_agent" not in config:
        raise ValueError(
            f"classification_agent not found in agents.yaml. "
            f"Available agents: {list(config.keys())}"
        )
    
    print("\n✓ classification_agent block found in agents.yaml")
    
    # Get the agent configuration
    agent_config = config["classification_agent"]
    print(f"\nclassification_agent configuration:")
    for key, value in agent_config.items():
        print(f"  {key}: {value}")
    
    # Verify prompt_file is set correctly
    expected_prompt_file = "classification.txt"
    actual_prompt_file = agent_config.get("prompt_file")
    
    if actual_prompt_file != expected_prompt_file:
        raise ValueError(
            f"Expected prompt_file to be '{expected_prompt_file}', "
            f"but got '{actual_prompt_file}'"
        )
    
    print(f"\n✓ prompt_file correctly points to: {actual_prompt_file}")
    
    # Get the absolute path to the prompt file
    prompts_dir = orchestrator_dir / "prompts"
    prompt_path = prompts_dir / actual_prompt_file
    
    print(f"\nPrompt file absolute path: {prompt_path.absolute()}")
    print(f"Prompt file exists: {prompt_path.exists()}")
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            f"Expected location: {prompts_dir}"
        )
    
    print(f"✓ Prompt file is accessible at: {prompt_path.absolute()}")
    
    # Verify other required fields
    required_fields = ["provider", "model", "temperature"]
    for field in required_fields:
        if field not in agent_config:
            raise ValueError(f"Missing required field: {field}")
        print(f"✓ {field}: {agent_config[field]}")
    
    print("\n" + "="*60)
    print("All checks passed! classification_agent is properly configured.")
    print("="*60)


if __name__ == "__main__":
    test_classification_agent_config()
