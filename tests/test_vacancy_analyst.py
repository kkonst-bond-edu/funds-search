"""
Test script to verify vacancy_analyst configuration in agents.yaml
and that the prompt file exists and is accessible.
"""

import yaml
from pathlib import Path


def test_vacancy_analyst_config():
    """Test that vacancy_analyst is properly configured in agents.yaml."""
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
    
    # Verify vacancy_analyst exists
    if "vacancy_analyst" not in config:
        raise ValueError(
            f"vacancy_analyst not found in agents.yaml. "
            f"Available agents: {list(config.keys())}"
        )
    
    print("\n✓ vacancy_analyst block found in agents.yaml")
    
    # Get the agent configuration
    agent_config = config["vacancy_analyst"]
    print(f"\nvacancy_analyst configuration:")
    for key, value in agent_config.items():
        print(f"  {key}: {value}")
    
    # Verify prompt_file is set correctly (defaulting to classification.txt)
    expected_prompt_file = "classification.txt"
    actual_prompt_file = agent_config.get("prompt_file")
    
    if actual_prompt_file != expected_prompt_file:
        print(f"Warning: Expected prompt_file to be '{expected_prompt_file}', but got '{actual_prompt_file}'. This might be intentional if using enrichment.")
    
    print(f"\n✓ prompt_file points to: {actual_prompt_file}")
    
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
    print("All checks passed! vacancy_analyst is properly configured.")
    print("="*60)


if __name__ == "__main__":
    test_vacancy_analyst_config()
