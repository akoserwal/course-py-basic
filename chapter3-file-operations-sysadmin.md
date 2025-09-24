# Chapter 3: File Operations and System Administration

## Learning Objectives
- Master file I/O operations for log processing and configuration management
- Work with various data formats (JSON, YAML, CSV) common in DevOps
- Understand OS module for system interactions
- Implement process management and monitoring
- Build practical tools for log analysis and system automation

## 3.1 File I/O Fundamentals

### Reading and Writing Files

```python
# Basic file operations
def read_config_file(filename):
    """Read configuration file with error handling."""
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Config file {filename} not found")
        return None
    except PermissionError:
        print(f"Permission denied reading {filename}")
        return None

def write_log_entry(filename, message):
    """Append log entry with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} - {message}\n"
    
    with open(filename, 'a') as file:
        file.write(log_entry)

# Reading large files efficiently
def process_large_log_file(filename):
    """Process large log files line by line."""
    error_count = 0
    warning_count = 0
    
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if 'ERROR' in line:
                error_count += 1
            elif 'WARNING' in line:
                warning_count += 1
            
            # Process every 1000 lines to show progress
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines...")
    
    return {
        'total_lines': line_num,
        'errors': error_count,
        'warnings': warning_count
    }
```

### File Path Operations

```python
from pathlib import Path
import os

def manage_log_directories():
    """Create and manage log directory structure."""
    # Using pathlib (modern approach)
    log_dir = Path("/var/log/myapp")
    
    # Create directories
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "errors").mkdir(exist_ok=True)
    (log_dir / "access").mkdir(exist_ok=True)
    (log_dir / "archived").mkdir(exist_ok=True)
    
    # Check if files exist
    error_log = log_dir / "errors" / "error.log"
    if error_log.exists():
        print(f"Error log size: {error_log.stat().st_size} bytes")
    
    # List log files
    log_files = list(log_dir.rglob("*.log"))
    return [str(f) for f in log_files]

def get_file_info(filepath):
    """Get comprehensive file information."""
    path = Path(filepath)
    
    if not path.exists():
        return None
    
    stat = path.stat()
    return {
        'name': path.name,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'permissions': oct(stat.st_mode)[-3:],
        'is_directory': path.is_dir(),
        'is_file': path.is_file(),
        'parent': str(path.parent),
        'absolute_path': str(path.absolute())
    }
```

## 3.2 Working with JSON

### Configuration Management

```python
import json
from typing import Dict, Any

class ConfigManager:
    """Manage JSON configuration files for applications."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found, using defaults")
            return self.default_config()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            return self.default_config()
    
    def default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "myapp"
            },
            "logging": {
                "level": "INFO",
                "file": "/var/log/myapp.log"
            }
        }
    
    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w') as file:
            json.dump(self.config, file, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

# Example usage
config = ConfigManager('app_config.json')
print(f"Server port: {config.get('server.port')}")
config.set('server.port', 9000)
config.save_config()
```

### API Response Processing

```python
import json
import requests
from typing import List, Dict

def process_api_responses(api_endpoints: List[str]) -> Dict[str, Any]:
    """Process multiple API endpoints and aggregate results."""
    results = {
        'healthy_services': [],
        'unhealthy_services': [],
        'total_response_time': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    for endpoint in api_endpoints:
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                service_info = {
                    'endpoint': endpoint,
                    'status': 'healthy',
                    'response_time': response_time,
                    'data': data
                }
                results['healthy_services'].append(service_info)
            else:
                results['unhealthy_services'].append({
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': response_time
                })
            
            results['total_response_time'] += response_time
            
        except requests.RequestException as e:
            results['unhealthy_services'].append({
                'endpoint': endpoint,
                'error': str(e)
            })
    
    return results
```

## 3.3 Working with YAML

### Infrastructure as Code Configuration

```python
import yaml
from typing import Dict, List, Any

class InfrastructureConfig:
    """Manage infrastructure configuration in YAML format."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(self.config_file, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self.create_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return {}
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default infrastructure configuration."""
        config = {
            'environments': {
                'development': {
                    'servers': [
                        {
                            'name': 'dev-web-01',
                            'type': 'web',
                            'cpu': 2,
                            'memory': '4GB',
                            'disk': '50GB'
                        }
                    ],
                    'databases': [
                        {
                            'name': 'dev-db-01', 
                            'type': 'postgresql',
                            'version': '14',
                            'storage': '100GB'
                        }
                    ]
                },
                'production': {
                    'servers': [
                        {
                            'name': 'prod-web-01',
                            'type': 'web',
                            'cpu': 8,
                            'memory': '16GB',
                            'disk': '200GB'
                        },
                        {
                            'name': 'prod-web-02',
                            'type': 'web', 
                            'cpu': 8,
                            'memory': '16GB',
                            'disk': '200GB'
                        }
                    ],
                    'databases': [
                        {
                            'name': 'prod-db-01',
                            'type': 'postgresql',
                            'version': '14',
                            'storage': '1TB',
                            'backup_enabled': True
                        }
                    ]
                }
            }
        }
        
        self.save_config(config)
        return config
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to YAML file."""
        config = config or self.config
        with open(self.config_file, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
    
    def get_servers(self, environment: str) -> List[Dict[str, Any]]:
        """Get servers for specific environment."""
        return self.config.get('environments', {}).get(environment, {}).get('servers', [])
    
    def add_server(self, environment: str, server_config: Dict[str, Any]):
        """Add server to environment."""
        if 'environments' not in self.config:
            self.config['environments'] = {}
        if environment not in self.config['environments']:
            self.config['environments'][environment] = {'servers': []}
        if 'servers' not in self.config['environments'][environment]:
            self.config['environments'][environment]['servers'] = []
        
        self.config['environments'][environment]['servers'].append(server_config)
        self.save_config()

# Example usage
infra = InfrastructureConfig('infrastructure.yaml')
prod_servers = infra.get_servers('production')
for server in prod_servers:
    print(f"Server: {server['name']} - CPU: {server['cpu']}, Memory: {server['memory']}")
```

### Ansible Integration

```python
def generate_ansible_inventory(infra_config: InfrastructureConfig) -> str:
    """Generate Ansible inventory from infrastructure configuration."""
    inventory = {
        'all': {
            'children': {}
        }
    }
    
    for env_name, env_config in infra_config.config['environments'].items():
        inventory['all']['children'][env_name] = {
            'children': {}
        }
        
        # Group servers by type
        server_types = {}
        for server in env_config.get('servers', []):
            server_type = server.get('type', 'unknown')
            if server_type not in server_types:
                server_types[server_type] = []
            server_types[server_type].append(server['name'])
        
        for server_type, hosts in server_types.items():
            group_name = f"{env_name}_{server_type}"
            inventory['all']['children'][env_name]['children'][group_name] = {
                'hosts': {host: {} for host in hosts}
            }
    
    return yaml.dump(inventory, default_flow_style=False)
```

## 3.4 Working with CSV Files

### System Metrics Processing

```python
import csv
from typing import List, Dict
from datetime import datetime, timedelta

class MetricsProcessor:
    """Process CSV files containing system metrics."""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
    
    def read_metrics(self) -> List[Dict[str, Any]]:
        """Read metrics from CSV file."""
        metrics = []
        
        try:
            with open(self.csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Convert numeric fields
                    metric = {
                        'timestamp': datetime.fromisoformat(row['timestamp']),
                        'hostname': row['hostname'],
                        'cpu_percent': float(row['cpu_percent']),
                        'memory_percent': float(row['memory_percent']),
                        'disk_percent': float(row['disk_percent']),
                        'network_in': int(row['network_in']),
                        'network_out': int(row['network_out'])
                    }
                    metrics.append(metric)
        
        except FileNotFoundError:
            print(f"Metrics file {self.csv_file} not found")
            return []
        
        return metrics
    
    def write_metrics(self, metrics: List[Dict[str, Any]]):
        """Write metrics to CSV file."""
        if not metrics:
            return
        
        fieldnames = ['timestamp', 'hostname', 'cpu_percent', 'memory_percent', 
                     'disk_percent', 'network_in', 'network_out']
        
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            for metric in metrics:
                # Convert datetime to string for CSV
                row = metric.copy()
                row['timestamp'] = metric['timestamp'].isoformat()
                writer.writerow(row)
    
    def analyze_metrics(self) -> Dict[str, Any]:
        """Analyze metrics and generate report."""
        metrics = self.read_metrics()
        if not metrics:
            return {}
        
        analysis = {
            'total_records': len(metrics),
            'time_range': {
                'start': min(m['timestamp'] for m in metrics),
                'end': max(m['timestamp'] for m in metrics)
            },
            'hosts': list(set(m['hostname'] for m in metrics)),
            'averages': {},
            'peaks': {},
            'alerts': []
        }
        
        # Calculate averages
        analysis['averages'] = {
            'cpu_percent': sum(m['cpu_percent'] for m in metrics) / len(metrics),
            'memory_percent': sum(m['memory_percent'] for m in metrics) / len(metrics),
            'disk_percent': sum(m['disk_percent'] for m in metrics) / len(metrics)
        }
        
        # Find peaks
        analysis['peaks'] = {
            'cpu_percent': max(m['cpu_percent'] for m in metrics),
            'memory_percent': max(m['memory_percent'] for m in metrics),
            'disk_percent': max(m['disk_percent'] for m in metrics)
        }
        
        # Generate alerts for high usage
        for metric in metrics:
            if metric['cpu_percent'] > 90:
                analysis['alerts'].append({
                    'type': 'HIGH_CPU',
                    'hostname': metric['hostname'],
                    'value': metric['cpu_percent'],
                    'timestamp': metric['timestamp']
                })
            
            if metric['memory_percent'] > 90:
                analysis['alerts'].append({
                    'type': 'HIGH_MEMORY',
                    'hostname': metric['hostname'],
                    'value': metric['memory_percent'],
                    'timestamp': metric['timestamp']
                })
        
        return analysis

def generate_sample_metrics():
    """Generate sample metrics data for testing."""
    import random
    
    metrics = []
    start_time = datetime.now() - timedelta(hours=24)
    
    hosts = ['web-01', 'web-02', 'db-01', 'cache-01']
    
    for i in range(1440):  # 24 hours of minute-by-minute data
        timestamp = start_time + timedelta(minutes=i)
        
        for host in hosts:
            metric = {
                'timestamp': timestamp,
                'hostname': host,
                'cpu_percent': random.uniform(10, 95),
                'memory_percent': random.uniform(30, 90),
                'disk_percent': random.uniform(40, 85),
                'network_in': random.randint(1000, 100000),
                'network_out': random.randint(1000, 50000)
            }
            metrics.append(metric)
    
    return metrics
```

## 3.5 OS Module and System Operations

### Process Management

```python
import os
import signal
import subprocess
import psutil
from typing import List, Dict, Optional

class ProcessManager:
    """Manage system processes and services."""
    
    @staticmethod
    def get_process_info(pid: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a process."""
        try:
            process = psutil.Process(pid)
            return {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'create_time': process.create_time(),
                'cmdline': process.cmdline(),
                'cwd': process.cwd(),
                'username': process.username()
            }
        except psutil.NoSuchProcess:
            return None
        except psutil.AccessDenied:
            return {'error': 'Access denied'}
    
    @staticmethod
    def find_processes_by_name(name: str) -> List[Dict[str, Any]]:
        """Find all processes with given name."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if name.lower() in proc.info['name'].lower():
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return processes
    
    @staticmethod
    def kill_process_tree(pid: int, timeout: int = 5) -> bool:
        """Kill a process and all its children."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=timeout)
            
            # Force kill remaining children
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Terminate parent
            parent.terminate()
            parent.wait(timeout=timeout)
            
            return True
            
        except psutil.NoSuchProcess:
            return False
        except psutil.TimeoutExpired:
            # Force kill parent
            try:
                parent.kill()
                return True
            except psutil.NoSuchProcess:
                return True
    
    @staticmethod
    def monitor_process_cpu(pid: int, interval: int = 1, duration: int = 60) -> List[float]:
        """Monitor CPU usage of a process over time."""
        cpu_usage = []
        
        try:
            process = psutil.Process(pid)
            
            for _ in range(duration):
                cpu_percent = process.cpu_percent(interval=interval)
                cpu_usage.append(cpu_percent)
                
                if not process.is_running():
                    break
            
        except psutil.NoSuchProcess:
            pass
        
        return cpu_usage

# Service management
class ServiceManager:
    """Manage system services using systemctl."""
    
    @staticmethod
    def is_service_active(service_name: str) -> bool:
        """Check if a service is active."""
        try:
            result = subprocess.run(['systemctl', 'is-active', service_name],
                                  capture_output=True, text=True)
            return result.stdout.strip() == 'active'
        except subprocess.SubprocessError:
            return False
    
    @staticmethod
    def get_service_status(service_name: str) -> Dict[str, Any]:
        """Get detailed service status."""
        try:
            result = subprocess.run(['systemctl', 'status', service_name],
                                  capture_output=True, text=True)
            
            return {
                'name': service_name,
                'active': ServiceManager.is_service_active(service_name),
                'status_output': result.stdout,
                'return_code': result.returncode
            }
        except subprocess.SubprocessError as e:
            return {
                'name': service_name,
                'error': str(e)
            }
    
    @staticmethod
    def restart_service(service_name: str) -> bool:
        """Restart a service."""
        try:
            result = subprocess.run(['sudo', 'systemctl', 'restart', service_name],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except subprocess.SubprocessError:
            return False
```

### System Information Gathering

```python
import platform
import socket
import shutil
from datetime import datetime

class SystemInfo:
    """Gather comprehensive system information."""
    
    @staticmethod
    def get_basic_info() -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def get_disk_usage() -> Dict[str, Dict[str, Any]]:
        """Get disk usage for all mounted filesystems."""
        disk_usage = {}
        
        # Get all disk partitions
        partitions = psutil.disk_partitions()
        
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'device': partition.device,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100
                }
            except PermissionError:
                disk_usage[partition.mountpoint] = {'error': 'Permission denied'}
        
        return disk_usage
    
    @staticmethod
    def get_network_interfaces() -> Dict[str, Dict[str, Any]]:
        """Get network interface information."""
        interfaces = {}
        
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        
        for interface_name, addresses in net_if_addrs.items():
            interface_info = {
                'addresses': [],
                'stats': net_if_stats.get(interface_name, {})._asdict() if interface_name in net_if_stats else {}
            }
            
            for addr in addresses:
                interface_info['addresses'].append({
                    'family': str(addr.family),
                    'address': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': addr.broadcast
                })
            
            interfaces[interface_name] = interface_info
        
        return interfaces
    
    @staticmethod
    def check_command_availability(commands: List[str]) -> Dict[str, bool]:
        """Check if commands are available in the system."""
        availability = {}
        
        for command in commands:
            availability[command] = shutil.which(command) is not None
        
        return availability
```

## Exercise 3: Build a Log Analyzer Tool

### Exercise Overview
Create a comprehensive log analyzer that processes web server logs, extracts meaningful insights, and generates reports in multiple formats.

### Step 1: Create the Project Structure

```bash
mkdir log-analyzer
cd log-analyzer
uv init
uv add click pyyaml tabulate matplotlib
mkdir -p {logs,config,reports,tests}
```

### Step 2: Create the Log Analyzer

Create `src/log_analyzer/analyzer.py`:

```python
#!/usr/bin/env python3
import re
import csv
import json
import yaml
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class LogAnalyzer:
    """Comprehensive log file analyzer for web servers."""
    
    def __init__(self, config_file: str = "config/analyzer.yaml"):
        self.config = self._load_config(config_file)
        self.logs = []
        self.analysis = {}
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load analyzer configuration."""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        default_config = {
            'log_format': 'apache_common',
            'time_window': 3600,  # 1 hour in seconds
            'alert_thresholds': {
                'error_rate': 5.0,  # percentage
                'response_time': 2000,  # milliseconds
                'requests_per_minute': 1000
            },
            'patterns': {
                'apache_common': r'(\S+) \S+ \S+ \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+|-)',
                'nginx': r'(\S+) - \S+ \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)"'
            }
        }
        
        # Save default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
        
        return default_config
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single log line based on configured format."""
        log_format = self.config['log_format']
        pattern = self.config['patterns'].get(log_format)
        
        if not pattern:
            return None
        
        match = re.match(pattern, line.strip())
        if not match:
            return None
        
        try:
            if log_format == 'apache_common':
                return {
                    'ip': match.group(1),
                    'timestamp': datetime.strptime(match.group(2), '%d/%b/%Y:%H:%M:%S %z'),
                    'method': match.group(3),
                    'path': match.group(4),
                    'protocol': match.group(5),
                    'status': int(match.group(6)),
                    'size': int(match.group(7)) if match.group(7) != '-' else 0
                }
            elif log_format == 'nginx':
                return {
                    'ip': match.group(1),
                    'timestamp': datetime.strptime(match.group(2), '%d/%b/%Y:%H:%M:%S %z'),
                    'method': match.group(3),
                    'path': match.group(4),
                    'protocol': match.group(5),
                    'status': int(match.group(6)),
                    'size': int(match.group(7)),
                    'referer': match.group(8),
                    'user_agent': match.group(9)
                }
        except (ValueError, IndexError):
            return None
        
        return None
    
    def load_log_file(self, filename: str) -> int:
        """Load and parse log file."""
        parsed_count = 0
        
        try:
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    parsed_entry = self.parse_log_line(line)
                    if parsed_entry:
                        self.logs.append(parsed_entry)
                        parsed_count += 1
                    
                    if line_num % 10000 == 0:
                        print(f"Processed {line_num} lines, parsed {parsed_count} entries")
        
        except FileNotFoundError:
            print(f"Log file {filename} not found")
            return 0
        
        print(f"Loaded {parsed_count} log entries from {filename}")
        return parsed_count
    
    def analyze_logs(self) -> Dict[str, Any]:
        """Perform comprehensive log analysis."""
        if not self.logs:
            return {}
        
        analysis = {
            'summary': self._analyze_summary(),
            'status_codes': self._analyze_status_codes(),
            'top_paths': self._analyze_top_paths(),
            'top_ips': self._analyze_top_ips(),
            'hourly_traffic': self._analyze_hourly_traffic(),
            'error_analysis': self._analyze_errors(),
            'security_alerts': self._analyze_security()
        }
        
        self.analysis = analysis
        return analysis
    
    def _analyze_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_requests = len(self.logs)
        if total_requests == 0:
            return {}
        
        timestamps = [log['timestamp'] for log in self.logs]
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()
        
        total_bytes = sum(log.get('size', 0) for log in self.logs)
        
        return {
            'total_requests': total_requests,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': duration / 3600
            },
            'requests_per_hour': total_requests / (duration / 3600) if duration > 0 else 0,
            'total_bytes': total_bytes,
            'average_request_size': total_bytes / total_requests
        }
    
    def _analyze_status_codes(self) -> Dict[str, Any]:
        """Analyze HTTP status code distribution."""
        status_counts = Counter(log['status'] for log in self.logs)
        total_requests = len(self.logs)
        
        return {
            'counts': dict(status_counts),
            'percentages': {
                status: (count / total_requests) * 100
                for status, count in status_counts.items()
            }
        }
    
    def _analyze_top_paths(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Analyze most requested paths."""
        path_counts = Counter(log['path'] for log in self.logs)
        
        return [
            {'path': path, 'count': count, 'percentage': (count / len(self.logs)) * 100}
            for path, count in path_counts.most_common(limit)
        ]
    
    def _analyze_top_ips(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Analyze top IP addresses."""
        ip_counts = Counter(log['ip'] for log in self.logs)
        
        return [
            {'ip': ip, 'count': count, 'percentage': (count / len(self.logs)) * 100}
            for ip, count in ip_counts.most_common(limit)
        ]
    
    def _analyze_hourly_traffic(self) -> Dict[str, int]:
        """Analyze traffic patterns by hour."""
        hourly_counts = defaultdict(int)
        
        for log in self.logs:
            hour = log['timestamp'].strftime('%Y-%m-%d %H:00')
            hourly_counts[hour] += 1
        
        return dict(hourly_counts)
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        error_logs = [log for log in self.logs if log['status'] >= 400]
        
        if not error_logs:
            return {'total_errors': 0}
        
        error_by_status = Counter(log['status'] for log in error_logs)
        error_by_path = Counter(log['path'] for log in error_logs)
        error_by_ip = Counter(log['ip'] for log in error_logs)
        
        return {
            'total_errors': len(error_logs),
            'error_rate': (len(error_logs) / len(self.logs)) * 100,
            'by_status': dict(error_by_status),
            'top_error_paths': [
                {'path': path, 'count': count}
                for path, count in error_by_path.most_common(10)
            ],
            'top_error_ips': [
                {'ip': ip, 'count': count}
                for ip, count in error_by_ip.most_common(10)
            ]
        }
    
    def _analyze_security(self) -> List[Dict[str, Any]]:
        """Analyze potential security threats."""
        alerts = []
        
        # Check for SQL injection attempts
        sql_patterns = [r'union.*select', r'drop.*table', r'insert.*into', r'delete.*from']
        for log in self.logs:
            path_lower = log['path'].lower()
            for pattern in sql_patterns:
                if re.search(pattern, path_lower):
                    alerts.append({
                        'type': 'SQL_INJECTION_ATTEMPT',
                        'ip': log['ip'],
                        'path': log['path'],
                        'timestamp': log['timestamp'].isoformat()
                    })
        
        # Check for suspicious IP activity
        ip_counts = Counter(log['ip'] for log in self.logs)
        threshold = self.config['alert_thresholds']['requests_per_minute'] * 60  # Convert to per hour
        
        for ip, count in ip_counts.items():
            if count > threshold:
                alerts.append({
                    'type': 'HIGH_REQUEST_VOLUME',
                    'ip': ip,
                    'request_count': count,
                    'threshold': threshold
                })
        
        return alerts
    
    def generate_report(self, format_type: str = 'text') -> str:
        """Generate analysis report in specified format."""
        if not self.analysis:
            return "No analysis data available. Run analyze_logs() first."
        
        if format_type == 'text':
            return self._generate_text_report()
        elif format_type == 'json':
            return json.dumps(self.analysis, indent=2, default=str)
        elif format_type == 'html':
            return self._generate_html_report()
        else:
            return "Unsupported format type"
    
    def _generate_text_report(self) -> str:
        """Generate text-based report."""
        lines = []
        lines.append("=" * 60)
        lines.append("LOG ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Summary
        summary = self.analysis['summary']
        lines.append(f"\nSUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total Requests: {summary['total_requests']:,}")
        lines.append(f"Time Range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        lines.append(f"Duration: {summary['time_range']['duration_hours']:.2f} hours")
        lines.append(f"Requests/Hour: {summary['requests_per_hour']:.2f}")
        lines.append(f"Total Bytes: {summary['total_bytes']:,}")
        
        # Status codes
        lines.append(f"\nSTATUS CODE DISTRIBUTION")
        lines.append("-" * 30)
        for status, percentage in self.analysis['status_codes']['percentages'].items():
            count = self.analysis['status_codes']['counts'][status]
            lines.append(f"  {status}: {count:,} ({percentage:.2f}%)")
        
        # Top paths
        lines.append(f"\nTOP REQUESTED PATHS")
        lines.append("-" * 25)
        for item in self.analysis['top_paths'][:10]:
            lines.append(f"  {item['path']}: {item['count']:,} ({item['percentage']:.2f}%)")
        
        # Errors
        error_analysis = self.analysis['error_analysis']
        lines.append(f"\nERROR ANALYSIS")
        lines.append("-" * 20)
        lines.append(f"Total Errors: {error_analysis['total_errors']:,}")
        lines.append(f"Error Rate: {error_analysis['error_rate']:.2f}%")
        
        # Security alerts
        alerts = self.analysis['security_alerts']
        if alerts:
            lines.append(f"\nSECURITY ALERTS")
            lines.append("-" * 20)
            for alert in alerts:
                lines.append(f"  {alert['type']}: {alert.get('ip', 'N/A')}")
        
        return "\n".join(lines)
    
    def save_report(self, filename: str, format_type: str = 'text'):
        """Save analysis report to file."""
        report = self.generate_report(format_type)
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filename}")

# Example usage and sample data generator
def generate_sample_logs(filename: str, num_entries: int = 10000):
    """Generate sample Apache log entries for testing."""
    import random
    from datetime import datetime, timedelta
    
    ips = ['192.168.1.100', '10.0.0.50', '203.0.113.25', '198.51.100.30']
    paths = ['/', '/index.html', '/api/users', '/api/data', '/login', '/admin', '/static/css/style.css']
    status_codes = [200, 200, 200, 200, 404, 500, 301, 403]
    methods = ['GET', 'POST', 'PUT', 'DELETE']
    
    start_time = datetime.now() - timedelta(hours=24)
    
    with open(filename, 'w') as f:
        for i in range(num_entries):
            timestamp = start_time + timedelta(seconds=random.randint(0, 86400))
            ip = random.choice(ips)
            method = random.choice(methods)
            path = random.choice(paths)
            status = random.choice(status_codes)
            size = random.randint(100, 5000)
            
            log_line = f'{ip} - - [{timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")}] "{method} {path} HTTP/1.1" {status} {size}\n'
            f.write(log_line)
    
    print(f"Generated {num_entries} sample log entries in {filename}")
```

### Step 3: Create CLI Interface

Create `src/log_analyzer/cli.py`:

```python
#!/usr/bin/env python3
import click
from pathlib import Path
from .analyzer import LogAnalyzer, generate_sample_logs

@click.group()
def cli():
    """Log Analyzer - Comprehensive web server log analysis tool."""
    pass

@cli.command()
@click.argument('log_file', type=click.Path(exists=True))
@click.option('--config', '-c', help='Configuration file path')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'html']), default='text')
@click.option('--output', '-o', help='Output file path')
def analyze(log_file, config, format, output):
    """Analyze log file and generate report."""
    click.echo(f"Analyzing log file: {log_file}")
    
    analyzer = LogAnalyzer(config) if config else LogAnalyzer()
    
    # Load log file
    count = analyzer.load_log_file(log_file)
    if count == 0:
        click.echo("No log entries found or parsed.")
        return
    
    # Perform analysis
    click.echo("Performing analysis...")
    analysis = analyzer.analyze_logs()
    
    # Generate report
    if output:
        analyzer.save_report(output, format)
        click.echo(f"Report saved to: {output}")
    else:
        report = analyzer.generate_report(format)
        click.echo(report)

@cli.command()
@click.argument('output_file')
@click.option('--entries', '-n', default=10000, help='Number of log entries to generate')
def generate(output_file, entries):
    """Generate sample log file for testing."""
    click.echo(f"Generating {entries} sample log entries...")
    generate_sample_logs(output_file, entries)
    click.echo(f"Sample log file created: {output_file}")

@cli.command()
@click.argument('log_file', type=click.Path(exists=True))
def summary(log_file):
    """Quick summary of log file."""
    analyzer = LogAnalyzer()
    count = analyzer.load_log_file(log_file)
    
    if count > 0:
        analysis = analyzer.analyze_logs()
        summary = analysis['summary']
        
        click.echo("Quick Summary:")
        click.echo(f"  Total Requests: {summary['total_requests']:,}")
        click.echo(f"  Duration: {summary['time_range']['duration_hours']:.2f} hours")
        click.echo(f"  Requests/Hour: {summary['requests_per_hour']:.2f}")
        click.echo(f"  Error Rate: {analysis['error_analysis']['error_rate']:.2f}%")

if __name__ == '__main__':
    cli()
```

### Step 4: Exercise Tasks

1. **Generate sample data:**
   ```bash
   uv run python -m log_analyzer.cli generate logs/sample.log --entries 50000
   ```

2. **Analyze logs:**
   ```bash
   uv run python -m log_analyzer.cli analyze logs/sample.log --format text
   ```

3. **Save reports:**
   ```bash
   uv run python -m log_analyzer.cli analyze logs/sample.log --format json --output reports/analysis.json
   ```

4. **Extend the analyzer:**
   - Add support for custom log formats
   - Implement real-time log monitoring
   - Add email alerts for security threats
   - Create visualization charts
   - Add database storage for historical analysis

## Key Takeaways

- File I/O operations are fundamental for log processing and configuration management
- JSON, YAML, and CSV are essential data formats in DevOps workflows
- The `os` and `pathlib` modules provide powerful system interaction capabilities
- Process management is crucial for service monitoring and automation
- Error handling ensures robust file operations
- Regular expressions enable complex log parsing
- Structured data analysis helps identify patterns and anomalies

This chapter provides the foundation for building sophisticated monitoring, analysis, and automation tools essential in SRE/DevOps roles.