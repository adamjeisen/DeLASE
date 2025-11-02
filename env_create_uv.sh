#!/bin/bash
set -e

echo "Creating Python environment with uv..."

# Parse optional demo extras flag
PACKAGE_SPEC="."
if [ "$1" = "--demo" ] || [ "$1" = "demo" ] || [ "$1" = "-d" ]; then
  echo "Including demo optional dependencies (extra: demo)"
  PACKAGE_SPEC=".[demo]"
fi

# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate
echo "Installing the project in editable mode..."
# Install the project in editable mode (optionally with demo extras)
uv pip install -e "$PACKAGE_SPEC"

echo "Installing ipykernel and creating a Jupyter kernel bound to this venv..."
# Add ipykernel as a dev dependency (persists to pyproject/uv.lock) and install the kernel
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name=DeLASE

echo "Environment created successfully!"
echo "To start Jupyter in Cursor/VSCode, click the kernel selecter in the top right corner."
echo "Then select the 'Select Another Kernel...' --> 'Jupyter Kernel...' --> 'DeLASE'"
echo "In the 'Jupyter Kernel...' selector, you may need to click the refresh button in the top right corner to see the new kernel."
echo "Note: If you do not see the new kernel, you may need to reload (i.e. close and reopen) your IDE window."
echo "To activate the environment, from inside this directory, run: source .venv/bin/activate"