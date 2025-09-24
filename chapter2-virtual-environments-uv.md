# Chapter 2: Virtual Environments with uv

## Learning Objectives
- Understand the importance of virtual environments in DevOps workflows
- Install and configure uv for Python project management
- Create and manage isolated Python environments
- Handle dependencies and project configurations
- Apply best practices for reproducible deployments

## 2.1 Why Virtual Environments Matter for SRE/DevOps

### The Problem
In SRE/DevOps work, you often manage multiple projects with different requirements:
- Monitoring tools requiring different versions of requests library
- Legacy applications using older Python versions
- New microservices with modern dependencies
- System packages that conflict with application requirements

### The Solution: Virtual Environments
Virtual environments provide:
- **Isolation**: Each project has its own dependencies
- **Reproducibility**: Exact dependency versions across environments
- **Safety**: System Python remains untouched
- **Portability**: Easy deployment and development setup

## 2.2 Introduction to uv

`uv` is a modern, fast Python package installer and resolver written in Rust. It's designed to be a drop-in replacement for pip and pip-tools with significant performance improvements.

### Why uv for DevOps?
- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Simplicity**: Easier project management
- **Modern**: Built for Python 3.7+ with modern standards

### Installation

```bash
# Install uv using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using Homebrew (macOS)
brew install uv

# Or using pip (if you have Python installed)
pip install uv

# Verify installation
uv --version
```

## 2.3 Basic uv Usage

### Creating Your First Virtual Environment

```bash
# Create a new directory for your project
mkdir monitoring-tools
cd monitoring-tools

# Initialize a new Python project with uv
uv init

# This creates:
# - pyproject.toml (project configuration)
# - .python-version (Python version specification)
# - src/monitoring_tools/ (source code directory)
# - README.md
```

### Understanding the Project Structure

```bash
monitoring-tools/
├── .python-version          # Specifies Python version
├── pyproject.toml           # Project metadata and dependencies
├── README.md
└── src/
    └── monitoring_tools/
        └── __init__.py
```

### Working with Dependencies

```bash
# Add a dependency
uv add requests

# Add development dependencies
uv add --dev pytest black flake8

# Add dependencies with version constraints
uv add "fastapi>=0.100.0,<1.0.0"
uv add "uvicorn[standard]"

# View your dependencies
uv tree

# Install all dependencies
uv sync
```

## 2.4 Project Configuration with pyproject.toml

### Understanding pyproject.toml
```toml
[project]
name = "monitoring-tools"
version = "0.1.0"
description = "SRE monitoring and automation tools"
authors = [{name = "Your Name", email = "your.email@company.com"}]
requires-python = ">=3.9"
dependencies = [
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
health-check = "monitoring_tools.health:main"
log-analyzer = "monitoring_tools.logs:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 88
target-version = ['py39']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### Advanced Dependency Management

```bash
# Add dependencies from requirements.txt
uv add -r requirements.txt

# Remove a dependency
uv remove requests

# Update all dependencies
uv sync --upgrade

# Lock dependencies (similar to pip freeze)
uv export --format requirements-txt > requirements.txt

# Install from lock file
uv sync --frozen
```

## 2.5 Managing Multiple Python Versions

### Installing Python Versions with uv

```bash
# Install specific Python version
uv python install 3.11

# List available Python versions
uv python list

# Set project Python version
uv python pin 3.11

# Use specific version for a project
echo "3.11" > .python-version
```

### Working with Different Projects

```bash
# Project 1: Legacy monitoring (Python 3.9)
mkdir legacy-monitor
cd legacy-monitor
uv init --python 3.9
uv add "requests==2.28.0" "pyyaml==5.4.1"

# Project 2: Modern API (Python 3.11)
cd ../
mkdir modern-api
cd modern-api
uv init --python 3.11
uv add "fastapi>=0.100.0" "uvicorn[standard]"
```

## 2.6 Running Commands in Virtual Environments

### Executing Python Code

```bash
# Run Python with project environment
uv run python script.py

# Run installed console scripts
uv run health-check

# Run modules
uv run -m pytest

# Start interactive Python shell
uv run python
```

### Running Development Tools

```bash
# Format code with black
uv run black src/

# Lint with flake8
uv run flake8 src/

# Type checking with mypy
uv run mypy src/

# Run tests
uv run pytest

# Combine multiple commands
uv run black src/ && uv run flake8 src/ && uv run pytest
```

## 2.7 Environment Variables and Configuration

### Setting Environment Variables

```bash
# Create .env file for development
cat > .env << EOF
API_KEY=dev-api-key-12345
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///dev.db
MONITORING_INTERVAL=30
EOF

# Load environment variables
uv run --env-file .env python app.py

# Set variables for specific commands
UV_ENV_VAR=value uv run python script.py
```

### Configuration for Different Environments

Create environment-specific files:
```bash
# .env.development
API_KEY=dev-key
LOG_LEVEL=DEBUG

# .env.staging  
API_KEY=staging-key
LOG_LEVEL=INFO

# .env.production
API_KEY=prod-key
LOG_LEVEL=WARNING
```

## Exercise 2: Setting up a Monitoring Project with uv

### Exercise Overview
Create a comprehensive monitoring project using uv that demonstrates best practices for dependency management and project structure.

### Step 1: Project Initialization

```bash
# Create the project
mkdir server-monitor
cd server-monitor
uv init --python 3.11

# Set up the project structure
mkdir -p {config,scripts,tests,docs}
mkdir -p src/server_monitor/{utils,collectors,alerts}
```

### Step 2: Add Dependencies

```bash
# Core dependencies
uv add requests pyyaml click psutil

# Development dependencies  
uv add --dev pytest pytest-cov black flake8 mypy

# Optional monitoring dependencies
uv add --optional monitoring "prometheus-client" "grafana-api"
```

### Step 3: Update pyproject.toml

Edit `pyproject.toml`:
```toml
[project]
name = "server-monitor"
version = "0.1.0"
description = "Server monitoring and alerting system"
authors = [{name = "SRE Team", email = "sre@company.com"}]
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
    "psutil>=5.9.0",
]

[project.optional-dependencies]
monitoring = [
    "prometheus-client>=0.17.0",
    "grafana-api>=1.0.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
monitor = "server_monitor.cli:main"
health-check = "server_monitor.health:check_health"

[tool.black]
line-length = 88
target-version = ['py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=src/server_monitor --cov-report=html"
```

### Step 4: Create Configuration Management

Create `src/server_monitor/config.py`:
```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv("CONFIG_FILE", "config/monitor.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(self.config_file)
        if not config_path.exists():
            return self._default_config()
        
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            "monitoring": {
                "interval": int(os.getenv("MONITORING_INTERVAL", "60")),
                "timeout": int(os.getenv("MONITORING_TIMEOUT", "30")),
            },
            "thresholds": {
                "cpu_warning": float(os.getenv("CPU_WARNING", "80")),
                "cpu_critical": float(os.getenv("CPU_CRITICAL", "95")),
                "memory_warning": float(os.getenv("MEMORY_WARNING", "85")),
                "memory_critical": float(os.getenv("MEMORY_CRITICAL", "95")),
            },
            "alerts": {
                "webhook_url": os.getenv("WEBHOOK_URL"),
                "email_enabled": os.getenv("EMAIL_ENABLED", "false").lower() == "true",
            }
        }
    
    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, {})
            if not isinstance(value, dict):
                return value
        return default or value
```

### Step 5: Create the Main CLI

Create `src/server_monitor/cli.py`:
```python
#!/usr/bin/env python3
import click
import time
from .config import Config
from .collectors.system import SystemCollector
from .alerts.manager import AlertManager

@click.group()
@click.option("--config", "-c", help="Configuration file path")
@click.pass_context
def main(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(config)

@main.command()
@click.option("--interval", "-i", type=int, help="Monitoring interval in seconds")
@click.pass_context
def monitor(ctx, interval):
    """Start continuous monitoring."""
    config = ctx.obj["config"]
    interval = interval or config.get("monitoring.interval", 60)
    
    collector = SystemCollector(config)
    alert_manager = AlertManager(config)
    
    click.echo(f"Starting monitoring with {interval}s interval...")
    
    try:
        while True:
            metrics = collector.collect()
            alerts = alert_manager.check_thresholds(metrics)
            
            if alerts:
                for alert in alerts:
                    click.echo(f"ALERT: {alert}")
                    alert_manager.send_alert(alert)
            else:
                click.echo(f"✓ All systems normal at {metrics['timestamp']}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped.")

@main.command()
@click.pass_context 
def check(ctx):
    """Run a single health check."""
    config = ctx.obj["config"]
    collector = SystemCollector(config)
    metrics = collector.collect()
    
    click.echo("System Health Check")
    click.echo("=" * 20)
    click.echo(f"CPU Usage: {metrics['cpu_percent']}%")
    click.echo(f"Memory Usage: {metrics['memory_percent']}%")
    click.echo(f"Disk Usage: {metrics['disk_percent']}%")

if __name__ == "__main__":
    main()
```

### Step 6: Create Configuration Files

Create `config/monitor.yaml`:
```yaml
monitoring:
  interval: 60
  timeout: 30

thresholds:
  cpu_warning: 80
  cpu_critical: 95
  memory_warning: 85
  memory_critical: 95
  disk_warning: 80
  disk_critical: 90

alerts:
  webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  email_enabled: false
  email_smtp_server: "smtp.company.com"
  email_recipients:
    - "sre-team@company.com"

servers:
  - name: "web-01"
    host: "192.168.1.10"
    port: 80
  - name: "db-01" 
    host: "192.168.1.20"
    port: 5432
```

### Step 7: Testing the Setup

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Run code formatting
uv run black src/

# Run linting
uv run flake8 src/

# Run the monitoring tool
uv run monitor check

# Start continuous monitoring
uv run monitor monitor --interval 30
```

### Step 8: Creating Environment-Specific Configurations

```bash
# Development environment
cat > .env.development << EOF
CONFIG_FILE=config/monitor.dev.yaml
LOG_LEVEL=DEBUG
MONITORING_INTERVAL=10
WEBHOOK_URL=https://hooks.slack.com/dev-webhook
EOF

# Production environment  
cat > .env.production << EOF
CONFIG_FILE=config/monitor.prod.yaml
LOG_LEVEL=WARNING
MONITORING_INTERVAL=60
WEBHOOK_URL=https://hooks.slack.com/prod-webhook
EOF

# Run with specific environment
uv run --env-file .env.development monitor check
```

## 2.8 Best Practices for DevOps Teams

### 1. Dependency Pinning Strategy
```bash
# Pin exact versions for production
uv add "requests==2.31.0"

# Use compatible versions for development
uv add "requests>=2.31.0,<3.0.0"

# Always commit lock files
git add uv.lock
```

### 2. Docker Integration
```dockerfile
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Run application
CMD ["uv", "run", "monitor", "monitor"]
```

### 3. CI/CD Integration
```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run tests
        run: uv run pytest
      - name: Run linting
        run: |
          uv run black --check src/
          uv run flake8 src/
```

### 4. Team Collaboration
```bash
# Share exact environment
uv export --format requirements-txt > requirements.lock

# Set up new developer environment
git clone repo
cd repo
uv sync

# Update dependencies across team
uv sync --upgrade
git add uv.lock
git commit -m "Update dependencies"
```

## Key Takeaways

- `uv` provides fast, reliable Python project management
- Virtual environments ensure reproducible deployments
- `pyproject.toml` centralizes project configuration
- Environment variables handle configuration differences
- Proper dependency management prevents conflicts
- Integration with Docker and CI/CD is straightforward

This foundation in virtual environments and dependency management is crucial for maintaining reliable, reproducible Python applications in production environments.