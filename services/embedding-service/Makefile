.PHONY: lint format test install clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run linting checks
lint:
	ruff check .
	black --check .

# Format code
format:
	black .
	ruff check --fix .

# Run tests
test:
	pytest

# Clean up Python cache
clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +

