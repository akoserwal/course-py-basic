# Chapter 7A: Basic Testing Fundamentals

## Learning Objectives
- Understand why testing matters in DevOps
- Write your first unit tests with pytest
- Learn test organization and best practices
- Create simple test automation workflows

## 7A.1 Why Test Everything?

In DevOps, testing is like having a safety net for your code:
- **Catch bugs early**: Find problems before they reach production
- **Confidence in changes**: Make changes without fear of breaking things
- **Documentation**: Tests show how your code should work
- **Automation**: Automatically verify everything works

Think of it as quality control in a factory - you test every part before shipping!

## 7A.2 Your First Unit Test

Let's start with the simplest possible test:

```python
# test_basics.py
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def test_add_numbers():
    """Test the add_numbers function."""
    result = add_numbers(2, 3)
    assert result == 5
```

### Running Your First Test
```bash
# Install pytest
pip install pytest

# Run the test
pytest test_basics.py -v
```

### Exercise 7A.1: Your First Test
1. Create `test_basics.py` with the code above
2. Run it and see it pass
3. Add a test for subtracting numbers
4. Make one test fail on purpose and see what happens

**Solution hint:**
```python
def subtract_numbers(a, b):
    return a - b

def test_subtract_numbers():
    result = subtract_numbers(5, 3)
    assert result == 2
```

## 7A.3 Testing a Simple Server Manager

Let's test something more realistic - a server management class:

```python
# server_manager.py
from datetime import datetime
from typing import Dict, List

class ServerManager:
    """Manage servers in our infrastructure."""
    
    def __init__(self):
        self.servers = {}
    
    def add_server(self, name: str, ip: str, port: int = 80) -> bool:
        """Add a server to our management system."""
        if not name or not ip:
            raise ValueError("Server name and IP are required")
        
        if name in self.servers:
            return False  # Server already exists
        
        self.servers[name] = {
            'ip': ip,
            'port': port,
            'status': 'active',
            'added_at': datetime.now()
        }
        return True
    
    def remove_server(self, name: str) -> bool:
        """Remove a server from management."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False
    
    def get_server(self, name: str) -> Dict:
        """Get server information."""
        return self.servers.get(name)
    
    def list_servers(self) -> List[str]:
        """List all server names."""
        return list(self.servers.keys())
    
    def count_servers(self) -> int:
        """Count total servers."""
        return len(self.servers)
```

Now let's test it:

```python
# test_server_manager.py
import pytest
from datetime import datetime
from server_manager import ServerManager

def test_server_manager_initialization():
    """Test that ServerManager starts empty."""
    manager = ServerManager()
    assert manager.count_servers() == 0
    assert manager.list_servers() == []

def test_add_server_success():
    """Test adding a server successfully."""
    manager = ServerManager()
    
    result = manager.add_server("web-01", "192.168.1.10", 80)
    
    assert result is True
    assert manager.count_servers() == 1
    assert "web-01" in manager.list_servers()
    
    # Check server details
    server = manager.get_server("web-01")
    assert server is not None
    assert server["ip"] == "192.168.1.10"
    assert server["port"] == 80
    assert server["status"] == "active"
    assert isinstance(server["added_at"], datetime)

def test_add_server_validation():
    """Test that server addition validates input."""
    manager = ServerManager()
    
    # Test empty name
    with pytest.raises(ValueError, match="Server name and IP are required"):
        manager.add_server("", "192.168.1.10")
    
    # Test empty IP
    with pytest.raises(ValueError, match="Server name and IP are required"):
        manager.add_server("web-01", "")

def test_add_duplicate_server():
    """Test that we can't add duplicate servers."""
    manager = ServerManager()
    
    # Add first server
    result1 = manager.add_server("web-01", "192.168.1.10")
    assert result1 is True
    
    # Try to add duplicate
    result2 = manager.add_server("web-01", "192.168.1.11")
    assert result2 is False
    
    # Should still have only one server
    assert manager.count_servers() == 1
    
    # Original server should be unchanged
    server = manager.get_server("web-01")
    assert server["ip"] == "192.168.1.10"

def test_remove_server():
    """Test removing a server."""
    manager = ServerManager()
    
    # Add server first
    manager.add_server("web-01", "192.168.1.10")
    assert manager.count_servers() == 1
    
    # Remove server
    result = manager.remove_server("web-01")
    
    assert result is True
    assert manager.count_servers() == 0
    assert "web-01" not in manager.list_servers()
    assert manager.get_server("web-01") is None

def test_remove_nonexistent_server():
    """Test removing a server that doesn't exist."""
    manager = ServerManager()
    
    result = manager.remove_server("nonexistent")
    assert result is False

def test_get_nonexistent_server():
    """Test getting a server that doesn't exist."""
    manager = ServerManager()
    
    server = manager.get_server("nonexistent")
    assert server is None

def test_multiple_servers():
    """Test managing multiple servers."""
    manager = ServerManager()
    
    # Add multiple servers
    servers = [
        ("web-01", "192.168.1.10", 80),
        ("web-02", "192.168.1.11", 80),
        ("db-01", "192.168.1.20", 5432)
    ]
    
    for name, ip, port in servers:
        result = manager.add_server(name, ip, port)
        assert result is True
    
    # Check count
    assert manager.count_servers() == 3
    
    # Check all servers exist
    server_names = manager.list_servers()
    assert "web-01" in server_names
    assert "web-02" in server_names
    assert "db-01" in server_names
    
    # Check individual servers
    db_server = manager.get_server("db-01")
    assert db_server["port"] == 5432
```

### Exercise 7A.2: Test a Server Manager
1. Create both `server_manager.py` and `test_server_manager.py`
2. Run the tests: `pytest test_server_manager.py -v`
3. Add a method to update server status and test it
4. Add a method to get servers by status and test it

## 7A.4 Test Organization with Fixtures

Fixtures help set up test data that multiple tests can use:

```python
# test_server_manager_with_fixtures.py
import pytest
from server_manager import ServerManager

@pytest.fixture
def empty_manager():
    """Create an empty ServerManager for testing."""
    return ServerManager()

@pytest.fixture
def populated_manager():
    """Create a ServerManager with sample data."""
    manager = ServerManager()
    
    # Add sample servers
    manager.add_server("web-01", "192.168.1.10", 80)
    manager.add_server("web-02", "192.168.1.11", 80)
    manager.add_server("db-01", "192.168.1.20", 5432)
    
    return manager

# Tests using fixtures
def test_empty_manager_fixture(empty_manager):
    """Test using the empty manager fixture."""
    assert empty_manager.count_servers() == 0

def test_populated_manager_fixture(populated_manager):
    """Test using the populated manager fixture."""
    assert populated_manager.count_servers() == 3
    assert "web-01" in populated_manager.list_servers()
    assert "db-01" in populated_manager.list_servers()

def test_add_server_to_populated_manager(populated_manager):
    """Test adding a server to an already populated manager."""
    initial_count = populated_manager.count_servers()
    
    result = populated_manager.add_server("cache-01", "192.168.1.30", 6379)
    
    assert result is True
    assert populated_manager.count_servers() == initial_count + 1
    assert "cache-01" in populated_manager.list_servers()

def test_remove_server_from_populated_manager(populated_manager):
    """Test removing a server from populated manager."""
    initial_count = populated_manager.count_servers()
    
    result = populated_manager.remove_server("web-01")
    
    assert result is True
    assert populated_manager.count_servers() == initial_count - 1
    assert "web-01" not in populated_manager.list_servers()
```

### Exercise 7A.3: Fixtures
1. Create `test_server_manager_with_fixtures.py`
2. Run the tests and see how fixtures work
3. Create a fixture with specific server configurations
4. Add tests that use your new fixture

## 7A.5 Parametrized Tests

Test the same function with different inputs:

```python
# test_parametrized.py
import pytest
from server_manager import ServerManager

# Test data for different server configurations
server_test_data = [
    ("web-01", "192.168.1.10", 80, True),     # Valid server
    ("web-02", "10.0.0.1", 8080, True),       # Valid server with different port
    ("db-01", "192.168.1.20", 5432, True),    # Valid database server
    ("", "192.168.1.10", 80, False),          # Empty name should fail
    ("web-01", "", 80, False),                # Empty IP should fail
]

@pytest.mark.parametrize("name,ip,port,should_succeed", server_test_data)
def test_add_server_parametrized(name, ip, port, should_succeed):
    """Test server addition with different parameters."""
    manager = ServerManager()
    
    if should_succeed:
        result = manager.add_server(name, ip, port)
        assert result is True
        assert name in manager.list_servers()
        
        server = manager.get_server(name)
        assert server["ip"] == ip
        assert server["port"] == port
    else:
        with pytest.raises(ValueError):
            manager.add_server(name, ip, port)

# Test different IP address formats
ip_test_data = [
    ("192.168.1.1", True),
    ("10.0.0.1", True),
    ("172.16.0.1", True),
    ("256.1.1.1", False),      # Invalid IP (too high)
    ("192.168.1", False),      # Incomplete IP
    ("not.an.ip", False),      # Not an IP
]

@pytest.mark.parametrize("ip,is_valid", ip_test_data)
def test_ip_validation(ip, is_valid):
    """Test IP address validation (simplified)."""
    import re
    
    # Simple IP pattern (for demo purposes)
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    
    if re.match(ip_pattern, ip):
        # Additional check for valid ranges (0-255)
        parts = ip.split('.')
        valid_parts = all(0 <= int(part) <= 255 for part in parts)
        assert is_valid == valid_parts
    else:
        assert not is_valid
```

### Exercise 7A.4: Parametrized Tests
1. Create `test_parametrized.py`
2. Run the tests: `pytest test_parametrized.py -v`
3. Add parametrized tests for port validation (1-65535)
4. Add tests for different server environments

## 7A.6 Simple Test Automation

Create a simple script to run tests automatically:

```python
# run_tests.py
#!/usr/bin/env python3
"""
Simple test runner script.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and report results."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        
        return False

def main():
    """Run all tests and checks."""
    print("🧪 Starting Test Automation")
    
    # Check if we're in the right directory
    if not Path("server_manager.py").exists():
        print("❌ server_manager.py not found. Run this from the correct directory.")
        sys.exit(1)
    
    tests_passed = 0
    total_tests = 0
    
    # Test commands to run
    test_commands = [
        ("pytest test_basics.py -v", "Basic Tests"),
        ("pytest test_server_manager.py -v", "Server Manager Tests"),
        ("pytest test_server_manager_with_fixtures.py -v", "Fixture Tests"),
        ("pytest test_parametrized.py -v", "Parametrized Tests"),
        ("pytest --tb=short", "All Tests (Short Output)"),
        ("pytest --cov=server_manager", "Tests with Coverage"),
    ]
    
    for command, description in test_commands:
        total_tests += 1
        if run_command(command, description):
            tests_passed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Exercise 7A.5: Test Automation
1. Create `run_tests.py` and make it executable
2. Install pytest-cov: `pip install pytest-cov`
3. Run the automation script: `python run_tests.py`
4. Add code quality checks (like checking for TODOs)

## 7A.7 Mini Project: Complete Testing Setup

Let's create a complete testing setup for a log analyzer:

```python
# log_analyzer.py
import re
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: datetime
    level: str
    message: str
    source: Optional[str] = None

class LogAnalyzer:
    """Analyze log files for patterns and errors."""
    
    def __init__(self):
        self.entries = []
        self.error_patterns = [
            r'error',
            r'exception',
            r'failed',
            r'timeout',
            r'connection refused'
        ]
    
    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line."""
        # Simple log pattern: 2024-01-01 12:00:00 [ERROR] Message
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
        
        match = re.match(pattern, line.strip())
        if not match:
            return None
        
        timestamp_str, level, message = match.groups()
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        return LogEntry(
            timestamp=timestamp,
            level=level.upper(),
            message=message
        )
    
    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the analyzer."""
        self.entries.append(entry)
    
    def parse_log_file(self, file_path: str):
        """Parse an entire log file."""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    entry = self.parse_log_line(line)
                    if entry:
                        self.add_log_entry(entry)
        except FileNotFoundError:
            raise ValueError(f"Log file not found: {file_path}")
    
    def get_entries_by_level(self, level: str) -> List[LogEntry]:
        """Get all entries of a specific level."""
        return [entry for entry in self.entries if entry.level == level.upper()]
    
    def get_error_count(self) -> int:
        """Count error-level entries."""
        return len(self.get_entries_by_level('ERROR'))
    
    def find_patterns(self, pattern: str) -> List[LogEntry]:
        """Find entries matching a specific pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [entry for entry in self.entries if regex.search(entry.message)]
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of log levels."""
        summary = {}
        for entry in self.entries:
            summary[entry.level] = summary.get(entry.level, 0) + 1
        return summary
```

Now let's create comprehensive tests:

```python
# test_log_analyzer.py
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from log_analyzer import LogAnalyzer, LogEntry

@pytest.fixture
def analyzer():
    """Create a LogAnalyzer instance."""
    return LogAnalyzer()

@pytest.fixture
def sample_log_file():
    """Create a temporary log file with sample data."""
    log_content = """2024-01-01 10:00:00 [INFO] Application started
2024-01-01 10:01:00 [DEBUG] Processing request
2024-01-01 10:02:00 [ERROR] Database connection failed
2024-01-01 10:03:00 [WARN] High memory usage detected
2024-01-01 10:04:00 [ERROR] Timeout occurred
2024-01-01 10:05:00 [INFO] Request completed
invalid log line without proper format
2024-01-01 10:06:00 [CRITICAL] System failure detected
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        f.write(log_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)

def test_log_entry_creation():
    """Test LogEntry creation."""
    timestamp = datetime(2024, 1, 1, 10, 0, 0)
    entry = LogEntry(
        timestamp=timestamp,
        level="ERROR",
        message="Test error message"
    )
    
    assert entry.timestamp == timestamp
    assert entry.level == "ERROR"
    assert entry.message == "Test error message"
    assert entry.source is None

def test_parse_log_line_valid(analyzer):
    """Test parsing a valid log line."""
    line = "2024-01-01 10:00:00 [ERROR] Database connection failed"
    
    entry = analyzer.parse_log_line(line)
    
    assert entry is not None
    assert entry.timestamp == datetime(2024, 1, 1, 10, 0, 0)
    assert entry.level == "ERROR"
    assert entry.message == "Database connection failed"

def test_parse_log_line_invalid(analyzer):
    """Test parsing an invalid log line."""
    line = "This is not a valid log line"
    
    entry = analyzer.parse_log_line(line)
    
    assert entry is None

def test_add_log_entry(analyzer):
    """Test adding log entries."""
    entry = LogEntry(
        timestamp=datetime.now(),
        level="INFO",
        message="Test message"
    )
    
    analyzer.add_log_entry(entry)
    
    assert len(analyzer.entries) == 1
    assert analyzer.entries[0] == entry

def test_parse_log_file(analyzer, sample_log_file):
    """Test parsing a complete log file."""
    analyzer.parse_log_file(sample_log_file)
    
    # Should have 7 valid entries (excluding the invalid line)
    assert len(analyzer.entries) == 7
    
    # Check first entry
    first_entry = analyzer.entries[0]
    assert first_entry.level == "INFO"
    assert first_entry.message == "Application started"

def test_parse_nonexistent_file(analyzer):
    """Test parsing a nonexistent file."""
    with pytest.raises(ValueError, match="Log file not found"):
        analyzer.parse_log_file("/nonexistent/file.log")

def test_get_entries_by_level(analyzer, sample_log_file):
    """Test getting entries by level."""
    analyzer.parse_log_file(sample_log_file)
    
    error_entries = analyzer.get_entries_by_level("ERROR")
    info_entries = analyzer.get_entries_by_level("INFO")
    
    assert len(error_entries) == 2
    assert len(info_entries) == 2
    
    # Check that all returned entries have the correct level
    for entry in error_entries:
        assert entry.level == "ERROR"

def test_get_error_count(analyzer, sample_log_file):
    """Test getting error count."""
    analyzer.parse_log_file(sample_log_file)
    
    error_count = analyzer.get_error_count()
    
    assert error_count == 2

def test_find_patterns(analyzer, sample_log_file):
    """Test finding entries by pattern."""
    analyzer.parse_log_file(sample_log_file)
    
    # Find entries with "connection" in the message
    connection_entries = analyzer.find_patterns("connection")
    
    assert len(connection_entries) == 1
    assert "connection failed" in connection_entries[0].message.lower()
    
    # Find entries with "system" (case insensitive)
    system_entries = analyzer.find_patterns("system")
    
    assert len(system_entries) == 1
    assert "system failure" in system_entries[0].message.lower()

def test_get_summary(analyzer, sample_log_file):
    """Test getting summary of log levels."""
    analyzer.parse_log_file(sample_log_file)
    
    summary = analyzer.get_summary()
    
    expected_summary = {
        'INFO': 2,
        'DEBUG': 1,
        'ERROR': 2,
        'WARN': 1,
        'CRITICAL': 1
    }
    
    assert summary == expected_summary

# Parametrized tests for different log formats
log_line_test_data = [
    ("2024-01-01 10:00:00 [INFO] Test", True, "INFO", "Test"),
    ("2024-12-31 23:59:59 [ERROR] Error occurred", True, "ERROR", "Error occurred"),
    ("2024-06-15 12:30:45 [DEBUG] Debug info", True, "DEBUG", "Debug info"),
    ("Invalid log line", False, None, None),
    ("2024-01-01 [INFO] Missing time", False, None, None),
    ("", False, None, None),
]

@pytest.mark.parametrize("line,should_parse,expected_level,expected_message", log_line_test_data)
def test_parse_log_line_parametrized(analyzer, line, should_parse, expected_level, expected_message):
    """Test log line parsing with different inputs."""
    entry = analyzer.parse_log_line(line)
    
    if should_parse:
        assert entry is not None
        assert entry.level == expected_level
        assert entry.message == expected_message
    else:
        assert entry is None
```

### Exercise 7A.6: Complete Testing Project
1. Create both `log_analyzer.py` and `test_log_analyzer.py`
2. Run all tests: `pytest test_log_analyzer.py -v --cov=log_analyzer`
3. Add tests for edge cases (empty files, large files)
4. Add a test for analyzing logs from multiple files
5. Create a simple test automation script

## Key Takeaways

- **Start simple**: Begin with basic assert statements
- **Use fixtures**: Reuse test setup across multiple tests
- **Test edge cases**: Empty inputs, invalid data, error conditions
- **Organize tests**: Group related tests and use clear naming
- **Automate testing**: Create scripts to run tests automatically
- **Use parametrized tests**: Test multiple scenarios efficiently

## Next Steps

In Chapter 7B, you'll learn about:
- Integration testing with APIs
- Testing with databases
- Advanced pytest features
- Continuous Integration setup
- Performance testing basics

Remember: Good tests make you confident in your code changes!