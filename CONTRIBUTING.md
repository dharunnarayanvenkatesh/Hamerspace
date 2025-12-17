# Contributing to Hamerspace

Thank you for your interest in contributing to Hamerspace! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/hamerspace.git
   cd hamerspace
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev,full]"
   ```

## Development Workflow

1. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Format your code**:
   ```bash
   black src/hamerspace
   isort src/hamerspace
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Add type hints where appropriate
- Write docstrings for all public functions and classes

## Testing

- Write tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)
- Use pytest for testing

## Adding New Backends

To add a new optimization backend:

1. Create a new file in `src/hamerspace/backends/`
2. Inherit from `BaseBackend`
3. Implement all required methods
4. Add tests in `tests/`
5. Update documentation

Example:
```python
from hamerspace.backends.base import BaseBackend

class MyBackend(BaseBackend):
    def __init__(self):
        super().__init__("MyBackend")
    
    def is_available(self) -> bool:
        # Check if dependencies are installed
        pass
    
    # Implement other required methods...
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all public APIs
- Include examples in `examples/` directory
- Update CHANGELOG.md

## Pull Request Guidelines

- **Title**: Clear and descriptive
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs as needed
- **Single Purpose**: One PR per feature/fix

## Reporting Issues

When reporting issues, please include:
- Hamerspace version
- Python version
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

## Feature Requests

We welcome feature requests! Please:
- Check if the feature already exists
- Explain the use case
- Describe the desired behavior
- Consider contributing the feature yourself

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment

## Questions?

- Open an issue with the "question" label
- Check existing issues and documentation first

## License

By contributing to Hamerspace, you agree that your contributions will be licensed under the Apache License 2.0.
