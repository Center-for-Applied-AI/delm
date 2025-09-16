# Releasing DELM to PyPI

This document explains how to release new versions of DELM to PyPI using GitHub Actions with trusted publishing.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org/account/register/)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Trusted Publishing**: Set up trusted publishing (required for this workflow)

## Setup (One-time)

### Trusted Publishing Setup

1. **Create PyPI Account**:
   - Go to [pypi.org/account/register/](https://pypi.org/account/register/)
   - Create account and verify your email
   - **Enable 2FA** (strongly recommended)

2. **Configure Trusted Publishing**:
   - Go to [PyPI Account Settings](https://pypi.org/manage/account/)
   - Navigate to "Publishing" → "Publishing projects"
   - Click "Add a new pending publisher"
   - Fill in:
     - **PyPI project name**: `delm`
     - **Owner**: Your GitHub username/organization
     - **Repository name**: `delm` (or your repo name)
     - **Workflow filename**: `publish.yml`
     - **Environment name**: Leave empty (uses default)
   - Click "Add"

3. **Verify Setup**:
   - The publisher will show as "pending" until you create your first release
   - Once you publish your first release, it will become "active"

## Release Process

### Method 1: Using the Release Script (Recommended)

```bash
# 1. Update version
python scripts/release.py 0.3.0

# 2. Commit and tag
git add .
git commit -m "Bump version to 0.3.0"
git tag v0.3.0
git push origin main --tags

# 3. Create GitHub release
# Go to GitHub → Releases → Create a new release
# Select tag v0.3.0 and publish
```

### Method 2: Manual Process

```bash
# 1. Update version in pyproject.toml and src/delm/__init__.py
# 2. Commit changes
git add .
git commit -m "Bump version to 0.3.0"

# 3. Create and push tag
git tag v0.3.0
git push origin main --tags

# 4. Create GitHub release
# Go to GitHub → Releases → Create a new release
# Select tag v0.3.0 and publish
```

## What Happens Next

1. **GitHub Actions triggers** when you create a release
2. **Tests run** on Python 3.8 and 3.11
3. **Package builds** if tests pass
4. **Package uploads** to PyPI automatically
5. **Package is available** at `pip install delm`

## Workflow Files

- `.github/workflows/test.yml` - Runs tests on push/PR
- `.github/workflows/publish.yml` - Publishes to PyPI on release

## Troubleshooting

### Common Issues

1. **"Package already exists"**: Version already published, increment version
2. **"Authentication failed"**: Check trusted publishing setup
3. **"Tests failing"**: Fix tests before releasing
4. **"Build failed"**: Check pyproject.toml syntax
5. **"Publisher not found"**: Verify trusted publishing configuration

### Trusted Publishing Issues

- **Publisher shows "pending"**: This is normal until first release
- **"Publisher not active"**: Check GitHub repository name and workflow filename
- **"Workflow not found"**: Ensure `.github/workflows/publish.yml` exists

### Manual Upload (Emergency)

If GitHub Actions fails, you can upload manually:

```bash
# Build package
python -m build

# Upload to PyPI (requires API token)
twine upload dist/*
```

## Version Management

- Use [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- Examples: `0.1.0`, `0.2.0`, `1.0.0`
- Update version in both `pyproject.toml` and `src/delm/__init__.py`

## Security Notes

- **Trusted publishing is secure** - No API tokens needed
- **Enable 2FA** on PyPI account
- **Review GitHub Actions logs** for security issues
- **Keep GitHub repository private** if needed during development
