# Chapter 1: Python Fundamentals for SRE/DevOps

## Learning Objectives
- Install Python and understand the development environment
- Master basic Python syntax essential for automation
- Work with data structures commonly used in DevOps tasks
- Create functions and modules for reusable code
- Build your first system administration script

## 1.1 Python Installation and Setup

### Installing Python
```bash
# On macOS (using Homebrew)
brew install python

# On Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# On CentOS/RHEL
sudo yum install python3 python3-pip
```

### Verify Installation
```bash
python3 --version
pip3 --version
```

### Setting up your development environment
```bash
# Install essential tools
pip3 install --user ipython black flake8

# Create a workspace directory
mkdir ~/devops-python
cd ~/devops-python
```

## 1.2 Basic Syntax and Data Types

### Variables and Basic Types
```python
# Variables (no declaration needed)
hostname = "web-server-01"
port = 8080
is_healthy = True
load_average = 2.5

# String operations (critical for log parsing)
log_line = "2024-01-15 10:30:45 ERROR: Connection failed"
timestamp = log_line[:19]  # Extract timestamp
level = log_line.split()[2].rstrip(':')  # Extract log level
print(f"Log level: {level} at {timestamp}")

# F-strings for formatting (essential for reports)
memory_usage = 75.5
alert_message = f"Memory usage is {memory_usage}% on {hostname}"
print(alert_message)
```

### Lists (for managing collections)
```python
# Server inventory
servers = ["web-01", "web-02", "db-01", "cache-01"]

# Adding servers
servers.append("web-03")
servers.extend(["db-02", "cache-02"])

# List comprehensions (powerful for filtering)
web_servers = [server for server in servers if server.startswith("web")]
print(f"Web servers: {web_servers}")

# Common operations
print(f"Total servers: {len(servers)}")
print(f"First server: {servers[0]}")
print(f"Last server: {servers[-1]}")
```

### Dictionaries (for configuration and metadata)
```python
# Server configuration
server_config = {
    "hostname": "web-01",
    "ip": "192.168.1.10",
    "port": 80,
    "status": "active",
    "tags": ["web", "production"]
}

# Accessing values
print(f"Server: {server_config['hostname']} at {server_config['ip']}")

# Safe access with get()
backup_port = server_config.get("backup_port", 8080)

# Updating configuration
server_config["last_check"] = "2024-01-15T10:30:00Z"
server_config.update({"cpu_cores": 4, "memory_gb": 16})

# Dictionary comprehension
healthy_servers = {name: config for name, config in servers_dict.items() 
                  if config.get("status") == "healthy"}
```

## 1.3 Control Structures

### Conditionals for Decision Making
```python
def check_server_status(cpu_usage, memory_usage, disk_usage):
    if cpu_usage > 90:
        return "CRITICAL: High CPU usage"
    elif memory_usage > 85:
        return "WARNING: High memory usage"
    elif disk_usage > 80:
        return "WARNING: High disk usage"
    else:
        return "OK: All systems normal"

# Usage
status = check_server_status(45, 70, 60)
print(status)
```

### Loops for Automation
```python
# Monitor multiple servers
servers = [
    {"name": "web-01", "cpu": 45, "memory": 70},
    {"name": "web-02", "cpu": 92, "memory": 60},
    {"name": "db-01", "cpu": 30, "memory": 88}
]

# Check each server
for server in servers:
    name = server["name"]
    cpu = server["cpu"]
    memory = server["memory"]
    
    if cpu > 90 or memory > 85:
        print(f"ALERT: {name} needs attention - CPU: {cpu}%, Memory: {memory}%")
    else:
        print(f"OK: {name} is healthy")

# Process log files
log_files = ["access.log", "error.log", "system.log"]
for log_file in log_files:
    print(f"Processing {log_file}...")
    # Would contain actual log processing logic
```

## 1.4 Functions and Modules

### Creating Reusable Functions
```python
def parse_log_line(line):
    """Parse a standard log line and return components."""
    try:
        parts = line.strip().split()
        timestamp = f"{parts[0]} {parts[1]}"
        level = parts[2].rstrip(':')
        message = ' '.join(parts[3:])
        
        return {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
    except IndexError:
        return None

def filter_logs_by_level(log_lines, level):
    """Filter log lines by severity level."""
    filtered = []
    for line in log_lines:
        parsed = parse_log_line(line)
        if parsed and parsed["level"] == level:
            filtered.append(parsed)
    return filtered

# Example usage
sample_logs = [
    "2024-01-15 10:30:45 INFO: Application started",
    "2024-01-15 10:31:20 ERROR: Database connection failed",
    "2024-01-15 10:31:45 INFO: Retrying connection",
    "2024-01-15 10:32:00 ERROR: Connection timeout"
]

errors = filter_logs_by_level(sample_logs, "ERROR")
for error in errors:
    print(f"Error at {error['timestamp']}: {error['message']}")
```

### Creating Modules
Create a file `server_utils.py`:
```python
import subprocess
import json
from datetime import datetime

def get_system_info():
    """Get basic system information."""
    try:
        # Get hostname
        hostname = subprocess.check_output(['hostname']).decode().strip()
        
        # Get uptime (simplified)
        uptime = subprocess.check_output(['uptime']).decode().strip()
        
        return {
            "hostname": hostname,
            "uptime": uptime,
            "timestamp": datetime.now().isoformat()
        }
    except subprocess.CalledProcessError:
        return {"error": "Failed to get system info"}

def ping_host(hostname):
    """Ping a host and return success status."""
    try:
        result = subprocess.run(['ping', '-c', '1', hostname], 
                               capture_output=True, timeout=5)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def format_bytes(bytes_value):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"
```

## 1.5 Working with Strings and Data Structures

### String Manipulation for Log Processing
```python
import re
from datetime import datetime

def extract_ip_addresses(log_content):
    """Extract all IP addresses from log content."""
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    return re.findall(ip_pattern, log_content)

def parse_apache_log(log_line):
    """Parse Apache common log format."""
    pattern = r'(\S+) \S+ \S+ \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+|-)'
    match = re.match(pattern, log_line)
    
    if match:
        return {
            "ip": match.group(1),
            "timestamp": match.group(2),
            "method": match.group(3),
            "path": match.group(4),
            "protocol": match.group(5),
            "status": int(match.group(6)),
            "size": match.group(7)
        }
    return None

# Example log processing
sample_apache_log = '192.168.1.100 - - [15/Jan/2024:10:30:45 +0000] "GET /health HTTP/1.1" 200 1234'
parsed = parse_apache_log(sample_apache_log)
if parsed:
    print(f"Request from {parsed['ip']} for {parsed['path']} returned {parsed['status']}")
```

## Exercise 1: System Information Script

Create a script that gathers and displays system information in a formatted way.

### Exercise Setup
Create a new file `system_info.py`:

```python
#!/usr/bin/env python3
"""
System Information Script for SRE/DevOps

This script gathers basic system information and presents it
in a formatted report suitable for monitoring dashboards.
"""

import subprocess
import platform
import psutil
import json
from datetime import datetime

def get_system_info():
    """Gather comprehensive system information."""
    info = {}
    
    # Basic system info
    info["hostname"] = platform.node()
    info["platform"] = platform.platform()
    info["python_version"] = platform.python_version()
    info["timestamp"] = datetime.now().isoformat()
    
    # CPU information
    info["cpu"] = {
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "usage_percent": psutil.cpu_percent(interval=1)
    }
    
    # Memory information
    memory = psutil.virtual_memory()
    info["memory"] = {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "usage_percent": memory.percent
    }
    
    # Disk information
    disk = psutil.disk_usage('/')
    info["disk"] = {
        "total_gb": round(disk.total / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "usage_percent": round((disk.used / disk.total) * 100, 2)
    }
    
    return info

def check_thresholds(info):
    """Check if any metrics exceed warning thresholds."""
    warnings = []
    
    if info["cpu"]["usage_percent"] > 80:
        warnings.append(f"High CPU usage: {info['cpu']['usage_percent']}%")
    
    if info["memory"]["usage_percent"] > 85:
        warnings.append(f"High memory usage: {info['memory']['usage_percent']}%")
    
    if info["disk"]["usage_percent"] > 90:
        warnings.append(f"High disk usage: {info['disk']['usage_percent']}%")
    
    return warnings

def format_report(info, warnings):
    """Format the information into a readable report."""
    report = []
    report.append("=" * 50)
    report.append(f"SYSTEM REPORT - {info['hostname']}")
    report.append(f"Generated: {info['timestamp']}")
    report.append("=" * 50)
    
    report.append(f"\nPlatform: {info['platform']}")
    report.append(f"Python: {info['python_version']}")
    
    report.append(f"\nCPU:")
    report.append(f"  Cores: {info['cpu']['cores']} (Threads: {info['cpu']['threads']})")
    report.append(f"  Usage: {info['cpu']['usage_percent']}%")
    
    report.append(f"\nMemory:")
    report.append(f"  Total: {info['memory']['total_gb']} GB")
    report.append(f"  Available: {info['memory']['available_gb']} GB")
    report.append(f"  Usage: {info['memory']['usage_percent']}%")
    
    report.append(f"\nDisk (/):")
    report.append(f"  Total: {info['disk']['total_gb']} GB")
    report.append(f"  Free: {info['disk']['free_gb']} GB")
    report.append(f"  Usage: {info['disk']['usage_percent']}%")
    
    if warnings:
        report.append(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            report.append(f"  - {warning}")
    else:
        report.append(f"\n✅ All systems normal")
    
    report.append("=" * 50)
    
    return "\n".join(report)

def main():
    """Main function to run the system info script."""
    try:
        # Gather system information
        info = get_system_info()
        
        # Check for warnings
        warnings = check_thresholds(info)
        
        # Generate report
        report = format_report(info, warnings)
        print(report)
        
        # Save to JSON file for programmatic access
        with open("system_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"\nDetailed info saved to system_info.json")
        
        # Return exit code based on warnings
        return 1 if warnings else 0
        
    except Exception as e:
        print(f"Error gathering system info: {e}")
        return 2

if __name__ == "__main__":
    exit(main())
```

### Exercise Tasks

1. **Install required dependencies:**
   ```bash
   pip3 install psutil
   ```

2. **Run the script:**
   ```bash
   python3 system_info.py
   ```

3. **Modify the script to:**
   - Add network interface information
   - Include running processes count
   - Add custom thresholds via command line arguments
   - Send alerts to a webhook when thresholds are exceeded

4. **Challenge:** Create a version that can monitor remote servers via SSH

### Key Takeaways

- Python's simplicity makes it ideal for system administration tasks
- String manipulation and regular expressions are crucial for log processing
- Data structures (lists, dictionaries) help organize infrastructure data
- Functions and modules promote code reuse and maintainability
- The `subprocess` module allows interaction with system commands
- Error handling is essential for robust automation scripts

This foundation prepares you for more advanced topics like API development, automation frameworks, and monitoring systems that you'll encounter in SRE/DevOps roles.