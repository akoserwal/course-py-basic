# Chapter 7: Testing and Automation

## Learning Objectives
- Master unit testing with pytest for reliable code
- Implement integration testing for APIs and services
- Create automated testing pipelines with CI/CD
- Build infrastructure automation scripts
- Design test-driven development workflows for DevOps
- Implement continuous testing and quality assurance

## 7.1 Unit Testing with pytest

### pytest Fundamentals

```python
# test_basic_operations.py
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

# Example module to test - server_manager.py
class ServerManager:
    """Manage server operations and health checks."""
    
    def __init__(self):
        self.servers = {}
        self.health_checks = {}
    
    def add_server(self, name: str, ip: str, port: int = 80) -> bool:
        """Add a server to management."""
        if not name or not ip:
            raise ValueError("Server name and IP are required")
        
        if name in self.servers:
            return False
        
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
            if name in self.health_checks:
                del self.health_checks[name]
            return True
        return False
    
    def get_server(self, name: str) -> Dict[str, Any]:
        """Get server information."""
        return self.servers.get(name)
    
    def update_health_status(self, name: str, is_healthy: bool) -> bool:
        """Update server health status."""
        if name not in self.servers:
            return False
        
        self.health_checks[name] = {
            'healthy': is_healthy,
            'last_check': datetime.now()
        }
        return True
    
    def get_healthy_servers(self) -> list:
        """Get list of healthy servers."""
        healthy = []
        for name, server in self.servers.items():
            health = self.health_checks.get(name, {})
            if health.get('healthy', True):  # Default to healthy if no check
                healthy.append(name)
        return healthy

# Basic test cases
def test_server_manager_initialization():
    """Test ServerManager initialization."""
    manager = ServerManager()
    assert manager.servers == {}
    assert manager.health_checks == {}

def test_add_server_success():
    """Test successful server addition."""
    manager = ServerManager()
    
    result = manager.add_server("web-01", "192.168.1.10", 80)
    
    assert result is True
    assert "web-01" in manager.servers
    assert manager.servers["web-01"]["ip"] == "192.168.1.10"
    assert manager.servers["web-01"]["port"] == 80
    assert manager.servers["web-01"]["status"] == "active"

def test_add_server_validation():
    """Test server addition validation."""
    manager = ServerManager()
    
    # Test empty name
    with pytest.raises(ValueError, match="Server name and IP are required"):
        manager.add_server("", "192.168.1.10")
    
    # Test empty IP
    with pytest.raises(ValueError, match="Server name and IP are required"):
        manager.add_server("web-01", "")

def test_add_duplicate_server():
    """Test adding duplicate server."""
    manager = ServerManager()
    
    # Add first server
    result1 = manager.add_server("web-01", "192.168.1.10")
    assert result1 is True
    
    # Try to add duplicate
    result2 = manager.add_server("web-01", "192.168.1.11")
    assert result2 is False
    
    # Original server should remain unchanged
    assert manager.servers["web-01"]["ip"] == "192.168.1.10"

def test_remove_server():
    """Test server removal."""
    manager = ServerManager()
    
    # Add server first
    manager.add_server("web-01", "192.168.1.10")
    manager.update_health_status("web-01", True)
    
    # Remove server
    result = manager.remove_server("web-01")
    
    assert result is True
    assert "web-01" not in manager.servers
    assert "web-01" not in manager.health_checks

def test_remove_nonexistent_server():
    """Test removing nonexistent server."""
    manager = ServerManager()
    
    result = manager.remove_server("nonexistent")
    assert result is False

def test_get_server():
    """Test getting server information."""
    manager = ServerManager()
    
    # Add server
    manager.add_server("web-01", "192.168.1.10", 8080)
    
    # Get server
    server = manager.get_server("web-01")
    
    assert server is not None
    assert server["ip"] == "192.168.1.10"
    assert server["port"] == 8080
    assert server["status"] == "active"
    assert isinstance(server["added_at"], datetime)

def test_get_nonexistent_server():
    """Test getting nonexistent server."""
    manager = ServerManager()
    
    server = manager.get_server("nonexistent")
    assert server is None

def test_update_health_status():
    """Test updating health status."""
    manager = ServerManager()
    
    # Add server
    manager.add_server("web-01", "192.168.1.10")
    
    # Update health status
    result = manager.update_health_status("web-01", True)
    
    assert result is True
    assert "web-01" in manager.health_checks
    assert manager.health_checks["web-01"]["healthy"] is True
    assert isinstance(manager.health_checks["web-01"]["last_check"], datetime)

def test_update_health_status_nonexistent():
    """Test updating health status for nonexistent server."""
    manager = ServerManager()
    
    result = manager.update_health_status("nonexistent", True)
    assert result is False

def test_get_healthy_servers():
    """Test getting list of healthy servers."""
    manager = ServerManager()
    
    # Add servers
    manager.add_server("web-01", "192.168.1.10")
    manager.add_server("web-02", "192.168.1.11") 
    manager.add_server("web-03", "192.168.1.12")
    
    # Set health status
    manager.update_health_status("web-01", True)
    manager.update_health_status("web-02", False)
    # web-03 has no health check (defaults to healthy)
    
    healthy = manager.get_healthy_servers()
    
    assert "web-01" in healthy
    assert "web-02" not in healthy
    assert "web-03" in healthy  # Defaults to healthy
```

### Fixtures and Test Setup

```python
# conftest.py - pytest configuration and fixtures
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def server_manager():
    """Create a ServerManager instance for testing."""
    return ServerManager()

@pytest.fixture
def populated_server_manager():
    """Create ServerManager with sample data."""
    manager = ServerManager()
    
    # Add sample servers
    manager.add_server("web-01", "192.168.1.10", 80)
    manager.add_server("web-02", "192.168.1.11", 80)
    manager.add_server("db-01", "192.168.1.20", 5432)
    
    # Set health status
    manager.update_health_status("web-01", True)
    manager.update_health_status("web-02", True)
    manager.update_health_status("db-01", False)
    
    return manager

@pytest.fixture
def sample_config_file(temp_directory):
    """Create sample configuration file."""
    config_content = """
servers:
  - name: web-01
    ip: 192.168.1.10
    port: 80
  - name: web-02
    ip: 192.168.1.11
    port: 80
monitoring:
  interval: 30
  timeout: 10
"""
    config_file = temp_directory / "config.yaml"
    config_file.write_text(config_content)
    return config_file

# Test using fixtures
def test_server_manager_fixture(server_manager):
    """Test using server_manager fixture."""
    assert len(server_manager.servers) == 0
    
    server_manager.add_server("test-server", "10.0.0.1")
    assert len(server_manager.servers) == 1

def test_populated_manager(populated_server_manager):
    """Test using populated server manager."""
    assert len(populated_server_manager.servers) == 3
    assert len(populated_server_manager.get_healthy_servers()) == 2

def test_config_file_loading(sample_config_file):
    """Test loading configuration from file."""
    import yaml
    
    with open(sample_config_file) as f:
        config = yaml.safe_load(f)
    
    assert len(config['servers']) == 2
    assert config['monitoring']['interval'] == 30
```

### Parametrized Tests

```python
import pytest

# Test data for parametrized tests
server_test_data = [
    ("web-01", "192.168.1.10", 80, True),
    ("web-02", "192.168.1.11", 8080, True),
    ("db-01", "192.168.1.20", 5432, True),
    ("", "192.168.1.10", 80, False),  # Empty name should fail
    ("web-01", "", 80, False),  # Empty IP should fail
]

@pytest.mark.parametrize("name,ip,port,should_succeed", server_test_data)
def test_add_server_parametrized(server_manager, name, ip, port, should_succeed):
    """Parametrized test for server addition."""
    if should_succeed:
        result = server_manager.add_server(name, ip, port)
        assert result is True
        assert name in server_manager.servers
        assert server_manager.servers[name]["ip"] == ip
        assert server_manager.servers[name]["port"] == port
    else:
        with pytest.raises(ValueError):
            server_manager.add_server(name, ip, port)

# Test IP address validation
ip_test_data = [
    ("192.168.1.1", True),
    ("10.0.0.1", True),
    ("255.255.255.255", True),
    ("192.168.1", False),
    ("256.1.1.1", False),
    ("not.an.ip", False),
    ("", False),
]

@pytest.mark.parametrize("ip,is_valid", ip_test_data)
def test_ip_validation(ip, is_valid):
    """Test IP address validation."""
    import ipaddress
    
    try:
        ipaddress.ip_address(ip)
        assert is_valid, f"Expected {ip} to be invalid"
    except ValueError:
        assert not is_valid, f"Expected {ip} to be valid"
```

### Mock and Patch Testing

```python
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

class ServiceHealthChecker:
    """Check health of external services."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def check_service(self, url: str) -> Dict[str, Any]:
        """Check service health via HTTP request."""
        try:
            response = requests.get(f"{url}/health", timeout=self.timeout)
            
            return {
                'url': url,
                'healthy': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'timestamp': datetime.now()
            }
        
        except requests.exceptions.RequestException as e:
            return {
                'url': url,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def check_multiple_services(self, urls: list) -> list:
        """Check health of multiple services."""
        results = []
        for url in urls:
            results.append(self.check_service(url))
        return results

# Tests using mocks
@patch('requests.get')
def test_service_health_check_success(mock_get):
    """Test successful health check using mock."""
    # Setup mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.elapsed.total_seconds.return_value = 0.5
    mock_get.return_value = mock_response
    
    # Test
    checker = ServiceHealthChecker()
    result = checker.check_service("http://example.com")
    
    # Assertions
    assert result['healthy'] is True
    assert result['status_code'] == 200
    assert result['response_time'] == 0.5
    assert result['url'] == "http://example.com"
    
    # Verify mock was called correctly
    mock_get.assert_called_once_with("http://example.com/health", timeout=30)

@patch('requests.get')
def test_service_health_check_failure(mock_get):
    """Test failed health check using mock."""
    # Setup mock to raise exception
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
    
    # Test
    checker = ServiceHealthChecker()
    result = checker.check_service("http://example.com")
    
    # Assertions
    assert result['healthy'] is False
    assert 'Connection failed' in result['error']
    assert result['url'] == "http://example.com"

@patch('requests.get')
def test_service_health_check_http_error(mock_get):
    """Test HTTP error response using mock."""
    # Setup mock response with error status
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.elapsed.total_seconds.return_value = 1.2
    mock_get.return_value = mock_response
    
    # Test
    checker = ServiceHealthChecker()
    result = checker.check_service("http://example.com")
    
    # Assertions
    assert result['healthy'] is False
    assert result['status_code'] == 500
    assert result['response_time'] == 1.2

@patch.object(ServiceHealthChecker, 'check_service')
def test_check_multiple_services(mock_check_service):
    """Test checking multiple services using mock."""
    # Setup mock to return different results
    mock_check_service.side_effect = [
        {'url': 'http://service1.com', 'healthy': True, 'status_code': 200},
        {'url': 'http://service2.com', 'healthy': False, 'status_code': 500},
        {'url': 'http://service3.com', 'healthy': True, 'status_code': 200},
    ]
    
    # Test
    checker = ServiceHealthChecker()
    urls = ['http://service1.com', 'http://service2.com', 'http://service3.com']
    results = checker.check_multiple_services(urls)
    
    # Assertions
    assert len(results) == 3
    assert results[0]['healthy'] is True
    assert results[1]['healthy'] is False
    assert results[2]['healthy'] is True
    
    # Verify mock was called for each URL
    assert mock_check_service.call_count == 3
```

## 7.2 Integration Testing

### API Integration Tests

```python
import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Assuming we have a FastAPI app from previous chapters
def create_test_app() -> FastAPI:
    """Create FastAPI app for testing."""
    from fastapi import FastAPI, HTTPException
    
    app = FastAPI()
    
    # In-memory storage for testing
    servers = {}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @app.get("/servers")
    async def list_servers():
        return {"servers": list(servers.values())}
    
    @app.post("/servers")
    async def create_server(server_data: dict):
        if not server_data.get("name") or not server_data.get("ip"):
            raise HTTPException(status_code=400, detail="Name and IP required")
        
        server_id = len(servers) + 1
        server = {
            "id": server_id,
            "name": server_data["name"],
            "ip": server_data["ip"],
            "port": server_data.get("port", 80),
            "status": "active"
        }
        servers[server_id] = server
        return server
    
    @app.get("/servers/{server_id}")
    async def get_server(server_id: int):
        if server_id not in servers:
            raise HTTPException(status_code=404, detail="Server not found")
        return servers[server_id]
    
    @app.delete("/servers/{server_id}")
    async def delete_server(server_id: int):
        if server_id not in servers:
            raise HTTPException(status_code=404, detail="Server not found")
        del servers[server_id]
        return {"message": "Server deleted"}
    
    return app

@pytest.fixture
def test_app():
    """Create test app fixture."""
    return create_test_app()

@pytest.fixture
def client(test_app):
    """Create test client fixture."""
    return TestClient(test_app)

# Integration tests
def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_create_and_get_server(client):
    """Test creating and retrieving a server."""
    # Create server
    server_data = {
        "name": "web-01",
        "ip": "192.168.1.10",
        "port": 80
    }
    
    create_response = client.post("/servers", json=server_data)
    
    assert create_response.status_code == 200
    created_server = create_response.json()
    assert created_server["name"] == "web-01"
    assert created_server["ip"] == "192.168.1.10"
    assert created_server["port"] == 80
    assert "id" in created_server
    
    # Get server
    server_id = created_server["id"]
    get_response = client.get(f"/servers/{server_id}")
    
    assert get_response.status_code == 200
    retrieved_server = get_response.json()
    assert retrieved_server == created_server

def test_create_server_validation(client):
    """Test server creation validation."""
    # Missing name
    response = client.post("/servers", json={"ip": "192.168.1.10"})
    assert response.status_code == 400
    
    # Missing IP
    response = client.post("/servers", json={"name": "web-01"})
    assert response.status_code == 400
    
    # Empty data
    response = client.post("/servers", json={})
    assert response.status_code == 400

def test_get_nonexistent_server(client):
    """Test getting nonexistent server."""
    response = client.get("/servers/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_delete_server(client):
    """Test deleting a server."""
    # Create server first
    server_data = {"name": "web-01", "ip": "192.168.1.10"}
    create_response = client.post("/servers", json=server_data)
    server_id = create_response.json()["id"]
    
    # Delete server
    delete_response = client.delete(f"/servers/{server_id}")
    
    assert delete_response.status_code == 200
    assert "deleted" in delete_response.json()["message"].lower()
    
    # Verify server is gone
    get_response = client.get(f"/servers/{server_id}")
    assert get_response.status_code == 404

def test_list_servers(client):
    """Test listing servers."""
    # Initially should be empty
    response = client.get("/servers")
    assert response.status_code == 200
    assert response.json()["servers"] == []
    
    # Add some servers
    for i in range(3):
        client.post("/servers", json={
            "name": f"web-{i:02d}",
            "ip": f"192.168.1.{10+i}"
        })
    
    # List servers
    response = client.get("/servers")
    assert response.status_code == 200
    servers = response.json()["servers"]
    assert len(servers) == 3
    
    # Verify server data
    server_names = [s["name"] for s in servers]
    assert "web-00" in server_names
    assert "web-01" in server_names
    assert "web-02" in server_names

# Async integration tests
@pytest.mark.asyncio
async def test_async_api_operations(test_app):
    """Test API operations using async client."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        # Test health check
        health_response = await ac.get("/health")
        assert health_response.status_code == 200
        
        # Create multiple servers concurrently
        server_data = [
            {"name": "web-01", "ip": "192.168.1.10"},
            {"name": "web-02", "ip": "192.168.1.11"},
            {"name": "db-01", "ip": "192.168.1.20", "port": 5432}
        ]
        
        create_tasks = [
            ac.post("/servers", json=data) for data in server_data
        ]
        
        create_responses = await asyncio.gather(*create_tasks)
        
        # Verify all servers were created
        for response in create_responses:
            assert response.status_code == 200
        
        # List all servers
        list_response = await ac.get("/servers")
        assert list_response.status_code == 200
        servers = list_response.json()["servers"]
        assert len(servers) == 3
```

### Database Integration Tests

```python
import pytest
import sqlite3
from pathlib import Path
import tempfile

class ServerDatabase:
    """Simple database for server management."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS servers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    ip TEXT NOT NULL,
                    port INTEGER DEFAULT 80,
                    status TEXT DEFAULT 'active',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_server(self, name: str, ip: str, port: int = 80) -> int:
        """Add server to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO servers (name, ip, port) VALUES (?, ?, ?)',
                (name, ip, port)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_server(self, server_id: int) -> dict:
        """Get server by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM servers WHERE id = ?',
                (server_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_servers(self) -> list:
        """List all servers."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM servers ORDER BY created_at')
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_server(self, server_id: int) -> bool:
        """Delete server by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM servers WHERE id = ?', (server_id,))
            conn.commit()
            return cursor.rowcount > 0

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)

@pytest.fixture
def server_db(temp_db):
    """Create ServerDatabase instance for testing."""
    return ServerDatabase(temp_db)

# Database integration tests
def test_database_initialization(temp_db):
    """Test database initialization."""
    db = ServerDatabase(temp_db)
    
    # Check that table was created
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='servers'"
        )
        assert cursor.fetchone() is not None

def test_add_and_get_server(server_db):
    """Test adding and retrieving server."""
    # Add server
    server_id = server_db.add_server("web-01", "192.168.1.10", 80)
    assert isinstance(server_id, int)
    assert server_id > 0
    
    # Get server
    server = server_db.get_server(server_id)
    assert server is not None
    assert server['name'] == "web-01"
    assert server['ip'] == "192.168.1.10"
    assert server['port'] == 80
    assert server['status'] == "active"
    assert server['id'] == server_id

def test_get_nonexistent_server(server_db):
    """Test getting nonexistent server."""
    server = server_db.get_server(999)
    assert server is None

def test_list_servers(server_db):
    """Test listing servers."""
    # Initially empty
    servers = server_db.list_servers()
    assert len(servers) == 0
    
    # Add servers
    server_data = [
        ("web-01", "192.168.1.10", 80),
        ("web-02", "192.168.1.11", 80),
        ("db-01", "192.168.1.20", 5432)
    ]
    
    for name, ip, port in server_data:
        server_db.add_server(name, ip, port)
    
    # List servers
    servers = server_db.list_servers()
    assert len(servers) == 3
    
    # Verify data
    server_names = [s['name'] for s in servers]
    assert "web-01" in server_names
    assert "web-02" in server_names
    assert "db-01" in server_names

def test_delete_server(server_db):
    """Test deleting server."""
    # Add server
    server_id = server_db.add_server("web-01", "192.168.1.10")
    
    # Delete server
    result = server_db.delete_server(server_id)
    assert result is True
    
    # Verify server is gone
    server = server_db.get_server(server_id)
    assert server is None

def test_delete_nonexistent_server(server_db):
    """Test deleting nonexistent server."""
    result = server_db.delete_server(999)
    assert result is False

def test_unique_constraint(server_db):
    """Test unique constraint on server name."""
    # Add first server
    server_db.add_server("web-01", "192.168.1.10")
    
    # Try to add duplicate name
    with pytest.raises(sqlite3.IntegrityError):
        server_db.add_server("web-01", "192.168.1.11")
```

## 7.3 CI/CD Pipeline Testing

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v1
    
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        uv sync --all-extras
    
    - name: Run linting
      run: |
        uv run black --check src/ tests/
        uv run flake8 src/ tests/
        uv run mypy src/
    
    - name: Run tests
      run: |
        uv run pytest tests/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
    
    - name: Run integration tests
      run: |
        uv run pytest tests/integration/ -v --tb=short
    
    - name: Run security checks
      run: |
        uv run bandit -r src/
        uv run safety check

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t myapp:${{ github.sha }} .
        docker tag myapp:${{ github.sha }} myapp:latest
    
    - name: Run container tests
      run: |
        docker run --rm myapp:${{ github.sha }} pytest tests/unit/
    
    - name: Push to registry
      if: success()
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push myapp:${{ github.sha }}
        docker push myapp:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Deploy to production
      run: |
        # Deployment commands here
        echo "Deploying to production..."
```

### Test Configuration Files

```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --disable-warnings
    -ra
    --cov=src
    --cov-branch
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    api: API tests
    database: Database tests

# pyproject.toml testing configuration
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### Makefile for Common Tasks

```makefile
# Makefile
.PHONY: help test test-unit test-integration test-coverage lint format install clean

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-coverage    Run tests with coverage"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean temporary files"

install:
	uv sync --all-extras

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-coverage:
	uv run pytest --cov=src --cov-report=html --cov-report=term

lint:
	uv run black --check src/ tests/
	uv run flake8 src/ tests/
	uv run mypy src/

format:
	uv run black src/ tests/
	uv run isort src/ tests/

security:
	uv run bandit -r src/
	uv run safety check

clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
```

## 7.4 Infrastructure Automation

### Automated Deployment Script

```python
#!/usr/bin/env python3
"""
Automated deployment script for DevOps applications.
"""

import os
import sys
import subprocess
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    service_name: str
    version: str
    environment: str
    replicas: int
    health_check_url: str
    rollback_enabled: bool = True
    timeout_seconds: int = 300

class DeploymentError(Exception):
    """Deployment-specific exception."""
    pass

class DeploymentManager:
    """Manage application deployments."""
    
    def __init__(self, config_file: str = "deploy.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.logger = self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if not Path(self.config_file).exists():
            raise DeploymentError(f"Config file {self.config_file} not found")
        
        with open(self.config_file) as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup deployment logging."""
        import logging
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging."""
        self.logger.info(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout:
                self.logger.info(f"Output: {result.stdout.strip()}")
            
            return result
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            if e.stderr:
                self.logger.error(f"Error: {e.stderr.strip()}")
            raise DeploymentError(f"Command failed: {command}")
    
    def build_image(self, deployment_config: DeploymentConfig) -> str:
        """Build Docker image."""
        self.logger.info(f"Building image for {deployment_config.service_name}:{deployment_config.version}")
        
        image_tag = f"{deployment_config.service_name}:{deployment_config.version}"
        
        # Build image
        self.run_command(f"docker build -t {image_tag} .")
        
        # Tag for registry
        registry_tag = f"{self.config['registry']['url']}/{image_tag}"
        self.run_command(f"docker tag {image_tag} {registry_tag}")
        
        return registry_tag
    
    def push_image(self, image_tag: str):
        """Push Docker image to registry."""
        self.logger.info(f"Pushing image: {image_tag}")
        
        # Login to registry
        registry_config = self.config['registry']
        self.run_command(
            f"echo {registry_config['password']} | docker login -u {registry_config['username']} --password-stdin {registry_config['url']}"
        )
        
        # Push image
        self.run_command(f"docker push {image_tag}")
    
    def deploy_to_kubernetes(self, deployment_config: DeploymentConfig, image_tag: str):
        """Deploy to Kubernetes cluster."""
        self.logger.info(f"Deploying to Kubernetes: {deployment_config.environment}")
        
        # Create deployment manifest
        manifest = self._create_k8s_manifest(deployment_config, image_tag)
        
        # Write manifest to file
        manifest_file = f"k8s-deployment-{deployment_config.service_name}.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f)
        
        # Apply manifest
        self.run_command(f"kubectl apply -f {manifest_file}")
        
        # Wait for rollout
        self.run_command(
            f"kubectl rollout status deployment/{deployment_config.service_name} --timeout={deployment_config.timeout_seconds}s"
        )
        
        # Cleanup
        Path(manifest_file).unlink()
    
    def _create_k8s_manifest(self, deployment_config: DeploymentConfig, image_tag: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_config.service_name,
                'namespace': deployment_config.environment,
                'labels': {
                    'app': deployment_config.service_name,
                    'version': deployment_config.version
                }
            },
            'spec': {
                'replicas': deployment_config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': deployment_config.service_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': deployment_config.service_name,
                            'version': deployment_config.version
                        }
                    },
                    'spec': {
                        'containers': [
                            {
                                'name': deployment_config.service_name,
                                'image': image_tag,
                                'ports': [
                                    {
                                        'containerPort': 8000
                                    }
                                ],
                                'livenessProbe': {
                                    'httpGet': {
                                        'path': '/health',
                                        'port': 8000
                                    },
                                    'initialDelaySeconds': 30,
                                    'periodSeconds': 10
                                },
                                'readinessProbe': {
                                    'httpGet': {
                                        'path': '/health',
                                        'port': 8000
                                    },
                                    'initialDelaySeconds': 5,
                                    'periodSeconds': 5
                                },
                                'resources': {
                                    'requests': {
                                        'cpu': '100m',
                                        'memory': '128Mi'
                                    },
                                    'limits': {
                                        'cpu': '500m',
                                        'memory': '512Mi'
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    
    def verify_deployment(self, deployment_config: DeploymentConfig) -> bool:
        """Verify deployment health."""
        self.logger.info("Verifying deployment health...")
        
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                import requests
                
                response = requests.get(
                    deployment_config.health_check_url,
                    timeout=10
                )
                
                if response.status_code == 200:
                    self.logger.info("Health check passed")
                    return True
                else:
                    self.logger.warning(f"Health check failed: HTTP {response.status_code}")
            
            except Exception as e:
                self.logger.warning(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(10)
        
        self.logger.error("Deployment verification failed")
        return False
    
    def rollback_deployment(self, deployment_config: DeploymentConfig):
        """Rollback deployment to previous version."""
        self.logger.info("Rolling back deployment...")
        
        self.run_command(
            f"kubectl rollout undo deployment/{deployment_config.service_name} -n {deployment_config.environment}"
        )
        
        self.run_command(
            f"kubectl rollout status deployment/{deployment_config.service_name} -n {deployment_config.environment} --timeout={deployment_config.timeout_seconds}s"
        )
    
    def deploy(self, deployment_config: DeploymentConfig):
        """Execute complete deployment process."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting deployment of {deployment_config.service_name}:{deployment_config.version}")
            
            # Build and push image
            image_tag = self.build_image(deployment_config)
            self.push_image(image_tag)
            
            # Deploy to Kubernetes
            self.deploy_to_kubernetes(deployment_config, image_tag)
            
            # Verify deployment
            if not self.verify_deployment(deployment_config):
                if deployment_config.rollback_enabled:
                    self.rollback_deployment(deployment_config)
                    raise DeploymentError("Deployment verification failed, rollback completed")
                else:
                    raise DeploymentError("Deployment verification failed")
            
            duration = time.time() - start_time
            self.logger.info(f"Deployment completed successfully in {duration:.2f} seconds")
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Deployment failed after {duration:.2f} seconds: {e}")
            raise

# Test the deployment manager
def test_deployment_manager():
    """Test deployment manager functionality."""
    # Create test configuration
    test_config = {
        'registry': {
            'url': 'localhost:5000',
            'username': 'test',
            'password': 'test'
        }
    }
    
    # Write test config
    with open('test_deploy.yaml', 'w') as f:
        yaml.dump(test_config, f)
    
    try:
        manager = DeploymentManager('test_deploy.yaml')
        
        # Test configuration loading
        assert manager.config['registry']['url'] == 'localhost:5000'
        
        # Test manifest creation
        deployment_config = DeploymentConfig(
            service_name='test-service',
            version='v1.0.0',
            environment='test',
            replicas=2,
            health_check_url='http://test-service/health'
        )
        
        manifest = manager._create_k8s_manifest(deployment_config, 'test-image:v1.0.0')
        
        assert manifest['metadata']['name'] == 'test-service'
        assert manifest['spec']['replicas'] == 2
        assert manifest['spec']['template']['spec']['containers'][0]['image'] == 'test-image:v1.0.0'
        
        print("All tests passed!")
    
    finally:
        # Cleanup
        Path('test_deploy.yaml').unlink(missing_ok=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_deployment_manager()
    else:
        # Example usage
        config = DeploymentConfig(
            service_name="my-api",
            version="v1.2.3",
            environment="production",
            replicas=3,
            health_check_url="https://my-api.example.com/health"
        )
        
        manager = DeploymentManager()
        manager.deploy(config)
```

## Exercise 7: Complete CI/CD Pipeline with Testing

### Exercise Overview
Create a complete CI/CD pipeline that includes comprehensive testing, automated deployment, and monitoring.

### Step 1: Project Structure

```bash
mkdir devops-pipeline-project
cd devops-pipeline-project
uv init
uv add "fastapi[all]" uvicorn pytest pytest-cov pytest-asyncio httpx black flake8 mypy bandit safety
mkdir -p {src/devops_app,tests/{unit,integration},scripts,.github/workflows,k8s}
```

### Step 2: Application Code

Create `src/devops_app/main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from datetime import datetime

app = FastAPI(title="DevOps Demo API", version="1.0.0")

# In-memory storage
services = {}
health_checks = {}

class ServiceCreate(BaseModel):
    name: str
    url: str
    environment: str

class ServiceResponse(BaseModel):
    id: str
    name: str
    url: str
    environment: str
    status: str
    created_at: datetime

@app.get("/")
async def root():
    return {"message": "DevOps Demo API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services_count": len(services)
    }

@app.post("/services", response_model=ServiceResponse)
async def create_service(service: ServiceCreate):
    service_id = f"service_{len(services) + 1}"
    
    service_data = {
        "id": service_id,
        "name": service.name,
        "url": service.url,
        "environment": service.environment,
        "status": "active",
        "created_at": datetime.now()
    }
    
    services[service_id] = service_data
    return ServiceResponse(**service_data)

@app.get("/services", response_model=List[ServiceResponse])
async def list_services():
    return [ServiceResponse(**service) for service in services.values()]

@app.get("/services/{service_id}", response_model=ServiceResponse)
async def get_service(service_id: str):
    if service_id not in services:
        raise HTTPException(status_code=404, detail="Service not found")
    return ServiceResponse(**services[service_id])

@app.delete("/services/{service_id}")
async def delete_service(service_id: str):
    if service_id not in services:
        raise HTTPException(status_code=404, detail="Service not found")
    del services[service_id]
    return {"message": "Service deleted"}

@app.get("/metrics")
async def metrics():
    return {
        "services_total": len(services),
        "uptime_seconds": time.time(),
        "timestamp": datetime.now().isoformat()
    }
```

### Step 3: Comprehensive Tests

Create `tests/unit/test_main.py`:

```python
import pytest
from fastapi.testclient import TestClient
from src.devops_app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "DevOps Demo API"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_service(client):
    service_data = {
        "name": "test-service",
        "url": "http://test.com",
        "environment": "test"
    }
    
    response = client.post("/services", json=service_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "test-service"
    assert data["url"] == "http://test.com"
    assert data["environment"] == "test"
    assert data["status"] == "active"
    assert "id" in data

def test_list_services(client):
    # Create services
    services = [
        {"name": "service-1", "url": "http://service1.com", "environment": "prod"},
        {"name": "service-2", "url": "http://service2.com", "environment": "dev"}
    ]
    
    for service in services:
        client.post("/services", json=service)
    
    # List services
    response = client.get("/services")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) == 2

def test_get_service(client):
    # Create service
    service_data = {"name": "test", "url": "http://test.com", "environment": "test"}
    create_response = client.post("/services", json=service_data)
    service_id = create_response.json()["id"]
    
    # Get service
    response = client.get(f"/services/{service_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "test"

def test_get_nonexistent_service(client):
    response = client.get("/services/nonexistent")
    assert response.status_code == 404

def test_delete_service(client):
    # Create service
    service_data = {"name": "test", "url": "http://test.com", "environment": "test"}
    create_response = client.post("/services", json=service_data)
    service_id = create_response.json()["id"]
    
    # Delete service
    response = client.delete(f"/services/{service_id}")
    assert response.status_code == 200
    
    # Verify deletion
    get_response = client.get(f"/services/{service_id}")
    assert get_response.status_code == 404

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "services_total" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data
```

### Step 4: Integration Tests

Create `tests/integration/test_api_integration.py`:

```python
import pytest
import asyncio
from httpx import AsyncClient
from src.devops_app.main import app

@pytest.mark.asyncio
async def test_full_service_lifecycle():
    """Test complete service lifecycle."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Check initial state
        response = await ac.get("/services")
        initial_count = len(response.json())
        
        # Create service
        service_data = {
            "name": "integration-test",
            "url": "http://integration.test",
            "environment": "test"
        }
        
        create_response = await ac.post("/services", json=service_data)
        assert create_response.status_code == 200
        service_id = create_response.json()["id"]
        
        # Verify service exists
        get_response = await ac.get(f"/services/{service_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == "integration-test"
        
        # Verify in list
        list_response = await ac.get("/services")
        assert len(list_response.json()) == initial_count + 1
        
        # Delete service
        delete_response = await ac.delete(f"/services/{service_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        final_response = await ac.get("/services")
        assert len(final_response.json()) == initial_count

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent API operations."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Create multiple services concurrently
        tasks = []
        for i in range(5):
            service_data = {
                "name": f"concurrent-service-{i}",
                "url": f"http://service{i}.test",
                "environment": "test"
            }
            tasks.append(ac.post("/services", json=service_data))
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Verify all services exist
        list_response = await ac.get("/services")
        services = list_response.json()
        
        service_names = [s["name"] for s in services]
        for i in range(5):
            assert f"concurrent-service-{i}" in service_names
```

### Step 5: CI/CD Configuration

Create `.github/workflows/cicd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: docker.io
  IMAGE_NAME: devops-demo-api

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.11]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v1
    
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: uv sync --all-extras
    
    - name: Run linting
      run: |
        uv run black --check src/ tests/
        uv run flake8 src/ tests/ --max-line-length=88
        uv run mypy src/
    
    - name: Run security checks
      run: |
        uv run bandit -r src/
        uv run safety check
    
    - name: Run unit tests
      run: |
        uv run pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        uv run pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to Kubernetes
      run: |
        export KUBECONFIG=kubeconfig
        export IMAGE_TAG="${{ needs.build.outputs.image-tag }}"
        envsubst < k8s/deployment.yaml | kubectl apply -f -
        kubectl rollout status deployment/devops-demo-api -n production --timeout=300s
    
    - name: Run smoke tests
      run: |
        # Wait for deployment
        sleep 30
        
        # Run smoke tests
        uv run pytest tests/smoke/ -v
```

### Step 6: Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uv", "run", "uvicorn", "src.devops_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 7: Kubernetes Manifests

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: devops-demo-api
  namespace: production
  labels:
    app: devops-demo-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: devops-demo-api
  template:
    metadata:
      labels:
        app: devops-demo-api
    spec:
      containers:
      - name: api
        image: ${IMAGE_TAG}
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        env:
        - name: ENVIRONMENT
          value: "production"
---
apiVersion: v1
kind: Service
metadata:
  name: devops-demo-api-service
  namespace: production
spec:
  selector:
    app: devops-demo-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Exercise Tasks

1. **Set up the project:**
   ```bash
   uv sync --all-extras
   ```

2. **Run tests locally:**
   ```bash
   make test
   make test-coverage
   make lint
   ```

3. **Build and test Docker image:**
   ```bash
   docker build -t devops-demo-api .
   docker run -p 8000:8000 devops-demo-api
   ```

4. **Set up GitHub repository and configure secrets**

5. **Push code and watch CI/CD pipeline execute**

6. **Advanced challenges:**
   - Add performance testing with locust
   - Implement blue-green deployments
   - Add monitoring with Prometheus
   - Create automated rollback triggers
   - Implement database migrations
   - Add end-to-end testing

## Key Takeaways

- Comprehensive testing ensures code reliability and prevents regressions
- Automated CI/CD pipelines improve deployment speed and consistency
- Infrastructure as code enables reproducible environments
- Monitoring and alerting are essential for production systems
- Test-driven development improves code quality and design
- Security scanning should be integrated into the development workflow

This complete testing and automation setup provides the foundation for reliable, scalable software delivery in modern DevOps environments.