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

## Before You Release

1. **Merge all changes to `main`**: All features and fixes for this release should already be merged via pull requests.

2. **Ensure tests pass**: The CI should be green on the `main` branch. Check [GitHub Actions](../../actions) to verify.

3. **Pull latest `main`**:
   ```bash
   git checkout main
   git pull origin main
   ```

4. **Review changes since last release**:
   ```bash
   git log $(git describe --tags --abbrev=0)..HEAD --oneline
   ```

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
2. **Tests run**
3. **Package builds** if tests pass
4. **Package uploads** to PyPI automatically
5. **Package is available** at `pip install delm`

## Workflow Files

- `.github/workflows/test.yml` - Runs tests on push/PR
- `.github/workflows/publish.yml` - Publishes to PyPI on release

## Version Management

- Use [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`
- Examples: `0.1.0`, `0.2.0`, `1.0.0`
- Update version in both `pyproject.toml` and `src/delm/__init__.py`

## Deploying Documentation

The documentation is built with MkDocs and deployed to GitHub Pages.

### Local Preview

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally with live reload
mkdocs serve

# View at http://127.0.0.1:8000
```

### Deploy to GitHub Pages

```bash
# Deploy docs to gh-pages branch
mkdocs gh-deploy
```

This command:
1. Builds the documentation from `docs/` and `mkdocs.yml`
2. Pushes to the `gh-pages` branch
3. GitHub Pages serves from that branch automatically

### When to Deploy Docs

- **After each release**: Deploy docs after publishing a new version to PyPI
- **After significant doc updates**: Deploy when documentation changes are merged to main

### Full Release Checklist

```bash
# 0. Ensure you're on main with all changes merged
git checkout main
git pull origin main

# 1. Update version
python scripts/release.py X.Y.Z

# 2. Commit and tag
git add .
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z

# 3. Push to main with tags
git push origin main --tags

# 4. Create GitHub release (triggers PyPI publish)
# Go to GitHub → Releases → Create new release → Select tag vX.Y.Z

# 5. Deploy docs (after PyPI publish succeeds)
mkdocs gh-deploy
```

## Security Notes

- **Trusted publishing is secure** - No API tokens needed
- **Enable 2FA** on PyPI account
- **Review GitHub Actions logs** for security issues
- **Keep GitHub repository private** if needed during development
