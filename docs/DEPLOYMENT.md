# Deployment Guide

This guide explains how to build, test, and publish Hamerspace to PyPI.

## Prerequisites

```bash
pip install build twine
```

## Building the Package

### 1. Update Version

Update version in:
- `setup.py`
- `pyproject.toml`
- `src/hamerspace/__init__.py`
- `CHANGELOG.md`

### 2. Build Distribution

```bash
# From the hamerspace root directory
python -m build
```

This creates:
- `dist/hamerspace-0.1.0.tar.gz` (source distribution)
- `dist/hamerspace-0.1.0-py3-none-any.whl` (wheel distribution)

### 3. Test the Build

```bash
# Install in a fresh environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate
pip install dist/hamerspace-0.1.0-py3-none-any.whl

# Test import
python -c "import hamerspace; print(hamerspace.__version__)"

# Run tests
pytest tests/
```

## Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ hamerspace
```

### Production PyPI

```bash
# Upload to production PyPI
twine upload dist/*

# Verify
pip install hamerspace
```

## GitHub Release

1. Create a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. Create GitHub release from the tag
3. Attach wheel and source distributions
4. Include release notes from CHANGELOG.md

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Version Management

Follow Semantic Versioning:
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Example:
- `0.1.0` - Initial release
- `0.2.0` - New backend added
- `0.2.1` - Bug fix
- `1.0.0` - Stable API

## Pre-release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Examples working
- [ ] README.md accurate
- [ ] License file present
- [ ] Contributors acknowledged

## Post-release Tasks

1. Announce on:
   - GitHub Discussions
   - Project blog/website
   - Social media
   - Relevant communities

2. Update documentation site

3. Monitor issues and feedback

4. Plan next release

## Troubleshooting

### Build Fails

- Check all dependencies are listed in `requirements.txt`
- Verify `setup.py` and `pyproject.toml` are correct
- Ensure all package files have `__init__.py`

### Upload Fails

- Verify PyPI credentials
- Check package name isn't taken
- Ensure version number is unique

### Installation Issues

- Test in clean virtual environment
- Check dependency versions
- Verify platform compatibility

## Maintenance

### Regular Updates

- Security patches for dependencies
- Bug fixes from issues
- Performance improvements
- New backend support

### Long-term Support

Consider LTS releases for stable API versions.
