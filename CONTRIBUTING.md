# Contributing to iceaggr

Welcome! This guide covers everything you need to contribute to IceCube neutrino direction reconstruction research. **Don't worry if you're new to this** - we'll walk through each step.

## 🎯 TL;DR - Quick Workflow

```bash
# 1. Get latest changes
git checkout main && git pull

# 2. Create your branch
git checkout -b experiment/my-cool-idea

# 3. Make changes, then test
uv run pytest
uv run ruff check .

# 4. Commit and push
git add . && git commit -m "Add cool idea"
git push origin experiment/my-cool-idea

# 5. Create Pull Request on GitHub
```

---

## 📋 Complete Setup Guide

### First Time Setup

#### 1. Install Required Tools

**Git** (if not already installed):
- Windows: Download from [git-scm.com](https://git-scm.com/)
- Mac: `xcode-select --install` or install via Homebrew
- Linux: `sudo apt install git` (Ubuntu) or equivalent

**UV** (Python package manager):
```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart terminal after installation
```

#### 2. Clone the Repository via SSH Access

- Generate SSH Key

Run this on the sever or your local machine where you want to use Git:
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```
where email is your GitHub one. Then it should create a private and a public key at ~/.ssh/id_ed25519 and ~/.ssh/id_ed25519.pub respectively.

- Copy the public key
```bash
cat ~/.ssh/id_ed25519.pub
```
- Add the key on the GitHub profile

Go to GitHub → Settings → SSH and GPG keys → New SSH key

Probably you'll be asked to approve it by your email for your key addition.
- Test your connection
```bash
ssh -T git@github.com
```
You should see something like this:
```bash
Hi <your-username>! You've successfully authenticated, but GitHub does not provide shell access.
```
- Clone the repository
Get the SSH URL from this GitHub repo
```bash
git clone git@github.com:timinar/iceaggr.git

cd iceaggr

# Setup everything automatically
uv sync
```

#### 3. Configure Git (First Time Only)

```bash
# Set your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@university.edu"

# Optional: Set VSCode as default editor
git config --global core.editor "code --wait"
```

#### 4. Setup VSCode (Recommended)

**Install VSCode** from [code.visualstudio.com](https://code.visualstudio.com/)

**Install Essential Extensions:**
1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Install these 3 extensions:
   - **Python** (by Microsoft)
   - **Ruff** (by Astral Software) 
   - **Jupyter** (by Microsoft)

**Configure VSCode:**
Create `.vscode/settings.json` in project root:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit"
    }
  },
  "notebook.formatOnSave.enabled": true
}
```

---

## 🔄 Daily Workflow

### Starting Work

**Always start with this:**
```bash
# Switch to main branch
git checkout main

# Get latest changes from team
git pull

# Sync dependencies (installs any new packages added by teammates)
uv sync
```

### Creating a New Experiment/Feature

#### 1. Create a Branch

Use descriptive names:
```bash
# For new experiments
git checkout -b experiment/transformer-attention

# For new features/tools
git checkout -b feature/data-loader-improvements

# For bug fixes
git checkout -b bugfix/memory-leak-training

# For analysis/visualization
git checkout -b analysis/loss-curves-investigation
```

#### 2. Work on Your Changes

**For Jupyter Notebooks:**
```bash
# Start Jupyter
uv run jupyter lab

# Work in notebooks/ directory
# Save frequently!
```

**For Python Code:**
```bash
# Open in VSCode
code .

# Work in src/iceaggr/ directory
# Run scripts with: uv run python scripts/your_script.py
```

**Adding Dependencies:**
```bash
# Need a new package? Add it with uv
uv add package-name

# For development-only tools (formatters, linters, etc.)
uv add --dev package-name

# Always commit pyproject.toml and uv.lock after adding packages
# Teammates will get the new packages when they run: uv sync
```

#### 3. Test Your Changes

**Run tests:**
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_models.py -v

# Run tests with coverage
uv run pytest --cov=src/iceaggr
```

**Check code quality:**
```bash
# Format code automatically
uv run ruff format .

# Check for issues
uv run ruff check .

# Fix automatically fixable issues
uv run ruff check . --fix
```

#### 4. Commit Your Changes

**Good commit messages:**
```bash
# ✅ Good commit messages
git commit -m "Add transformer model with multi-head attention"
git commit -m "Fix memory leak in data loader"
git commit -m "Update dataset preprocessing for new physics data"

# ❌ Avoid these
git commit -m "stuff"
git commit -m "fix"
git commit -m "changes"
```

**Commit workflow:**
```bash
# Add all changes
git add .

# Or add specific files
git add src/hep_ssl/models/transformer.py notebooks/transformer_experiments.ipynb

# Commit with descriptive message
git commit -m "Add transformer model with attention mechanism for particle sequence modeling"

# Push to your branch
git push origin experiment/transformer-attention
```

### Creating a Pull Request

#### 1. Push Your Branch
```bash
git push origin your-branch-name
```

#### 2. Create PR on GitHub
1. Go to the repository on GitHub
2. Click "Compare & pull request" button
3. Fill out the template:

```markdown
## What does this PR do?
Brief description of your changes

## Key changes
- Added transformer model with attention
- Updated training script to support new model
- Added tests for transformer components

## Testing
- [ ] All tests pass
- [ ] Code is formatted with ruff
- [ ] Tested on sample data

## Experiment Results (if applicable)
- Training time: X minutes
- Performance: Y% accuracy
- W&B run: [link to experiment]
```

#### 3. Address Review Comments
When team members review:
```bash
# Make requested changes
# Then commit and push
git add .
git commit -m "Address review comments: fix docstrings"
git push origin your-branch-name
```

---

## 🧪 Experiment Management

### Running Experiments

#### Basic Training
```bash
# Simple training run
uv run python scripts/train_model.py

# With specific config (future)
uv run python scripts/train_model.py --config configs/transformer.yaml
```

#### Jupyter Notebook Experiments
```bash
# Start Jupyter Lab
uv run jupyter lab

# Name notebooks descriptively:
# 01-data-exploration.ipynb
# 02-transformer-baseline.ipynb  
# 03-diffusion-model-experiments.ipynb
```

### Weights & Biases Integration

#### First Time Setup
```bash
# Login to W&B
uv run wandb login
# Follow the instructions, paste your API key
```

#### In Your Code
```python
import wandb

# Start experiment tracking
wandb.init(
    project="iceaggr",
    name="dom-transformer-v1",
    tags=["dom-level", "baseline", "your-name"]
)

# Log metrics during training
wandb.log({"loss": loss, "angular_error": angular_err})

# Log models/artifacts
wandb.log_artifact("model.pt")
```

#### Experiment Naming Convention
- **Project**: `iceaggr`
- **Run names**: `component-description-version-yourname`
  - `dom-transformer-baseline-v1-alice`
  - `event-transformer-geom-v2-bob`
  - `dataloader-continuous-batch-v1-charlie`
- **Tags**: `["component", "experiment-type", "your-name"]`
  - DOM-level experiments: `["dom-level", ...]`
  - Event-level experiments: `["event-level", ...]`
  - End-to-end: `["e2e", ...]`

---

## 📊 Code Quality & Best Practices

### Python Code Style

**We use Ruff for formatting and linting:**
```bash
# Auto-format your code
uv run ruff format .

# Check for issues
uv run ruff check .

# Fix automatically
uv run ruff check . --fix
```

**Good practices:**
```python
# ✅ Good
def train_model(model, data_loader, epochs: int = 10):
    """Train the model on provided data.
    
    Args:
        model: The PyTorch model to train
        data_loader: Training data loader
        epochs: Number of training epochs
    """
    for epoch in range(epochs):
        # training logic here
        pass

# ❌ Avoid
def train(m,d,e):
    for i in range(e):
        pass
```

### Jupyter Notebooks

**Good notebook practices:**
- Clear markdown explanations before code blocks
- Remove old/unused cells before committing
- Restart kernel and run all cells before sharing
- Save outputs if they're important results

**Notebook structure:**
```markdown
# Experiment: Transformer Attention Analysis
## Goal
Investigate different attention mechanisms...

## Setup
```

```python
import torch
import matplotlib.pyplot as plt
# ... imports
```

```markdown
## Data Loading
```

```python
# Load and preprocess data
```

```markdown
## Model Training
```

### Testing & Benchmarking

**Write tests for important functions:**
```python
# tests/unit/test_models.py
import torch
from iceaggr.models.dom_transformer import DOMTransformer

def test_dom_transformer_forward():
    """Test DOM transformer handles variable-length sequences."""
    model = DOMTransformer(d_model=128, n_heads=8)
    batch = create_mock_batch()

    output, metadata = model(batch)

    assert output.shape[1] == 128  # d_model dimension
    assert not torch.isnan(output).any()
```

**Two types of performance testing:**

**1. Regression Tests** (`tests/benchmarks/`) - Run via pytest:
```python
# tests/benchmarks/test_dataloader_performance.py
def test_dataloader_throughput():
    """Ensure dataloader maintains >100 events/sec."""
    loader = get_dataloader(batch_size=32)
    throughput = measure_throughput(loader)
    assert throughput > 100, f"Too slow: {throughput:.1f} events/sec"
```

**2. Exploratory Benchmarks** (`scripts/`) - Standalone scripts:
```bash
# Deep dive into performance characteristics
uv run python scripts/benchmark_dom_packing.py

# Generate plots/reports for analysis
uv run python scripts/analyze_memory_scaling.py
```

**When to use which:**
- **Use `tests/benchmarks/`** for: CI/CD gates, regression prevention, quick checks
- **Use `scripts/`** for: Finding bottlenecks, scaling analysis, report generation

---

## 🐛 Common Issues & Solutions

### UV/Python Issues

**"UV command not found":**
```bash
# Restart terminal, or manually source
source $HOME/.local/bin/env

# On Windows, restart PowerShell/Command Prompt
```

**"Wrong Python version":**
```bash
# Check what UV is using
uv python list

# Install specific version
uv python install 3.13
uv python pin 3.13
```

### Git Issues

**"I messed up my branch":**
```bash
# Save your work
git stash

# Reset to clean state
git checkout main
git pull
git checkout -b new-branch-name

# Get your work back
git stash pop
```

**"I committed to main by mistake":**
```bash
# DON'T PUSH! Create new branch first
git checkout -b fix/my-changes

# Then reset main
git checkout main
git reset --hard HEAD~1  # Undo last commit
```

**"Merge conflicts":**
```bash
# Pull latest main
git checkout main && git pull

# Merge main into your branch
git checkout your-branch
git merge main

# Fix conflicts in VSCode (it highlights them)
# Then commit the merge
git add . && git commit -m "Resolve merge conflicts"
```

### Code Issues

**"Import errors":**
```bash
# Make sure you're in the right directory and using uv run
cd hep_ssl
uv run python scripts/your_script.py

# For imports in notebooks, restart kernel
```

**"CUDA out of memory":**
```python
# Reduce batch size
batch_size = 16  # instead of 64

# Clear cache
torch.cuda.empty_cache()

# Use gradient checkpointing (Lightning)
model = YourModel(gradient_checkpointing=True)
```

---

## 📞 Getting Help

### Before Asking for Help

1. **Check this guide** - solution might be here
2. **Google the error message** - often quick fixes exist
3. **Check if others had the same issue** in our GitHub issues

### When to Ask for Help

- You've been stuck for >30 minutes
- Error messages are unclear
- You're not sure about the approach
- You want feedback on experimental design

### How to Ask for Help

**Good help request:**
```
Hi! I'm trying to implement attention in the transformer model, but getting this error:

RuntimeError: Expected tensor to have same device as self (cuda:0), but got cpu

Here's my code: [paste relevant code]
What I've tried: moved model to GPU with .cuda()
Environment: PyTorch 2.0, RTX 3080

Any ideas?
```

**Include:**
- What you're trying to do
- The exact error message
- Relevant code snippet
- What you've already tried
- Your setup (GPU, PyTorch version, etc.)

---

## 🎉 You're Ready!

This might seem like a lot, but don't worry:
1. **Start small** - make a tiny change, commit, push
2. **Ask questions** - we're here to help
3. **Learn by doing** - you'll pick it up quickly
4. **Don't be afraid to experiment** - that's what research is about!

Remember: **Perfect is the enemy of good**. It's better to make progress and iterate than to wait for the perfect solution.

Welcome to the team! 🚀⚡