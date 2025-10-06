# Contributing to Anomalous Market Behavior Recognition

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Basic understanding of machine learning and time series analysis

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/jamesolaitan/Anomalous-Market-Behavior-Recognition-with-Machine-Learning.git
cd Anomalous-Market-Behavior-Recognition-with-Machine-Learning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (including dev dependencies)
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📝 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/updates

### 2. Make Changes

- Write clean, readable code
- Follow existing code style
- Add docstrings to functions and classes
- Update tests as needed
- Update documentation if applicable

### 3. Format Code

```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking (optional but recommended)
mypy src/ --ignore-missing-imports
```

Or use the Makefile:

```bash
make format  # Format code
make lint    # Check linting
```

### 4. Run Tests

```bash
# Run all tests
make test

# Or manually
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add support for additional symbols"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Test additions/updates
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Detailed description of what and why
- Reference any related issues
- Screenshots/examples if applicable

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

Example:

```python
def test_compute_rolling_correlation_with_valid_data():
    """Test that rolling correlation is computed correctly with valid input."""
    # Arrange
    df = create_sample_data()
    
    # Act
    result = compute_rolling_correlation(df, window=30)
    
    # Assert
    assert "rolling_corr" in result.columns
    assert not result["rolling_corr"].isna().all()
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Test both success and failure paths

## 📚 Documentation Guidelines

### Docstrings

Use Google-style docstrings:

```python
def compute_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling features from price data.

    Args:
        df: DataFrame with price data
        window: Rolling window size in days

    Returns:
        DataFrame with computed features

    Raises:
        ValueError: If window is less than 1
    """
    if window < 1:
        raise ValueError("Window must be at least 1")
    # ... implementation
```

### README Updates

If your change affects usage:
- Update relevant sections in README.md
- Add examples if introducing new features
- Update configuration documentation

## 🐛 Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, dependency versions
6. **Logs**: Relevant error messages or logs

Example:

```markdown
**Description**: Model training fails with NaN loss

**Steps to Reproduce**:
1. Run `make train`
2. Observe loss becomes NaN after epoch 5

**Expected**: Training should complete successfully

**Actual**: Training fails with NaN loss

**Environment**:
- OS: macOS 14.0
- Python: 3.10.9
- PyTorch: 2.0.1

**Logs**:
```
Epoch 5/100 - Train Loss: nan
```
```

## 💡 Feature Requests

When requesting features:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Suggest how it could work
3. **Alternatives**: Other approaches you've considered
4. **Impact**: Who would benefit from this feature

## 🔍 Code Review Process

### For Contributors

- Respond to feedback promptly
- Be open to suggestions
- Ask questions if unclear
- Update based on review comments

### For Reviewers

- Be constructive and specific
- Explain the "why" behind suggestions
- Approve when ready or request changes
- Test the changes locally if possible

## 📋 Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] No unnecessary files included

## 🎯 Areas for Contribution

Looking for ideas? Consider:

### Easy Wins
- Improve documentation
- Add more tests
- Fix typos
- Add examples

### Medium Difficulty
- Add new features (e.g., additional symbols, time periods)
- Improve visualizations
- Optimize performance
- Add new evaluation metrics

### Advanced
- Implement alternative models (Transformer, GRU)
- Add hyperparameter tuning
- Implement online learning
- Add streaming data support

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email directly (see README)

## 🙏 Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Credited in release notes
- Mentioned in README (for significant contributions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉

