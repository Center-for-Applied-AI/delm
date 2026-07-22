#!/usr/bin/env python3
"""
Release script for DELM package.
Usage: python scripts/release.py <version>
Example: python scripts/release.py 0.3.0
"""

import sys
import re
from pathlib import Path


def update_version(version: str):
    """Update version in pyproject.toml and __init__.py"""

    # Update pyproject.toml. The pattern is anchored to the start of the line
    # so it cannot match other keys such as mypy's python_version.
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    content = re.sub(
        r'^version = "[^"]*"',
        f'version = "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    pyproject_path.write_text(content)
    print(f"✅ Updated pyproject.toml to version {version}")

    # Update __init__.py
    init_path = Path("src/delm/__init__.py")
    content = init_path.read_text()
    content = re.sub(r'__version__ = "[^"]*"', f'__version__ = "{version}"', content)
    init_path.write_text(content)
    print(f"✅ Updated src/delm/__init__.py to version {version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/release.py <version>")
        print("Example: python scripts/release.py 0.3.0")
        sys.exit(1)

    version = sys.argv[1]

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        print("❌ Invalid version format. Use semantic versioning (e.g., 0.3.0)")
        sys.exit(1)

    print(f"🚀 Preparing release {version}...")
    update_version(version)
    print(f"✅ Version {version} updated successfully!")
    print("\nNext steps:")
    print("1. git add .")
    print(f"2. git commit -m 'Bump version to {version}'")
    print(f"3. git tag v{version}")
    print("4. git push origin main --tags")
    print("5. Create a GitHub release with tag v{version}")


if __name__ == "__main__":
    main()
