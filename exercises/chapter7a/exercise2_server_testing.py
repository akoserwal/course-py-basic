#!/usr/bin/env python3
"""
Exercise 7A.2: Testing a Server Manager
Learn to test a more complex class with multiple methods.
"""

from datetime import datetime
from typing import Dict, List

# Server Manager class to test
class ServerManager:
    """Manage servers in our infrastructure."""
    
    def __init__(self):
        self.servers = {}
        self.health_checks = {}
    
    def add_server(self, name: str, ip: str, port: int = 80) -> bool:
        """Add a server to our management system."""
        # TODO: Add validation
        # Uncomment and complete these validations:
        #
        # if not name or not ip:
        #     raise ValueError("Server name and IP are required")
        
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
        # TODO: Implement server removal
        # Remove from both self.servers and self.health_checks if they exist
        # Return True if server was removed, False if it didn't exist
        pass
    
    def get_server(self, name: str) -> Dict:
        """Get server information."""
        # TODO: Implement getting server info
        # Return the server dict or None if not found
        pass
    
    def list_servers(self) -> List[str]:
        """List all server names."""
        # TODO: Return a list of all server names
        pass
    
    def count_servers(self) -> int:
        """Count total servers."""
        # TODO: Return the number of servers
        pass
    
    def update_server_status(self, name: str, status: str) -> bool:
        """Update server status."""
        # TODO: Implement status update
        # Valid statuses: active, maintenance, stopped
        # Return True if updated, False if server doesn't exist
        pass
    
    def get_servers_by_status(self, status: str) -> List[str]:
        """Get servers with a specific status."""
        # TODO: Return list of server names with the given status
        pass
    
    def add_health_check(self, name: str, is_healthy: bool) -> bool:
        """Add a health check result for a server."""
        # TODO: Implement health check recording
        # Store health status and timestamp
        # Return True if successful, False if server doesn't exist
        pass

# TODO: Complete all the test functions below

def test_server_manager_initialization():
    """Test that ServerManager starts empty."""
    manager = ServerManager()
    
    # TODO: Test that manager starts with no servers
    # assert manager.count_servers() == 0
    # assert manager.list_servers() == []
    pass

def test_add_server_success():
    """Test adding a server successfully."""
    manager = ServerManager()
    
    # TODO: Test adding a server
    # result = manager.add_server("web-01", "192.168.1.10", 80)
    # assert result is True
    # assert manager.count_servers() == 1
    # assert "web-01" in manager.list_servers()
    
    # TODO: Check server details
    # server = manager.get_server("web-01")
    # assert server is not None
    # assert server["ip"] == "192.168.1.10"
    # assert server["port"] == 80
    # assert server["status"] == "active"
    # assert isinstance(server["added_at"], datetime)
    pass

def test_add_server_validation():
    """Test that server addition validates input."""
    # TODO: Import pytest for exception testing
    # import pytest
    
    manager = ServerManager()
    
    # TODO: Test empty name should raise ValueError
    # with pytest.raises(ValueError, match="Server name and IP are required"):
    #     manager.add_server("", "192.168.1.10")
    
    # TODO: Test empty IP should raise ValueError
    # with pytest.raises(ValueError, match="Server name and IP are required"):
    #     manager.add_server("web-01", "")
    pass

def test_add_duplicate_server():
    """Test that we can't add duplicate servers."""
    manager = ServerManager()
    
    # TODO: Add first server and verify success
    # result1 = manager.add_server("web-01", "192.168.1.10")
    # assert result1 is True
    
    # TODO: Try to add duplicate and verify failure
    # result2 = manager.add_server("web-01", "192.168.1.11")
    # assert result2 is False
    
    # TODO: Verify still only one server exists
    # assert manager.count_servers() == 1
    
    # TODO: Verify original server unchanged
    # server = manager.get_server("web-01")
    # assert server["ip"] == "192.168.1.10"
    pass

def test_remove_server():
    """Test removing a server."""
    manager = ServerManager()
    
    # TODO: Add server first
    # manager.add_server("web-01", "192.168.1.10")
    # assert manager.count_servers() == 1
    
    # TODO: Remove server
    # result = manager.remove_server("web-01")
    # assert result is True
    # assert manager.count_servers() == 0
    # assert "web-01" not in manager.list_servers()
    # assert manager.get_server("web-01") is None
    pass

def test_remove_nonexistent_server():
    """Test removing a server that doesn't exist."""
    manager = ServerManager()
    
    # TODO: Try to remove nonexistent server
    # result = manager.remove_server("nonexistent")
    # assert result is False
    pass

def test_get_nonexistent_server():
    """Test getting a server that doesn't exist."""
    manager = ServerManager()
    
    # TODO: Try to get nonexistent server
    # server = manager.get_server("nonexistent")
    # assert server is None
    pass

def test_update_server_status():
    """Test updating server status."""
    manager = ServerManager()
    
    # TODO: Add server first
    # manager.add_server("web-01", "192.168.1.10")
    
    # TODO: Update status
    # result = manager.update_server_status("web-01", "maintenance")
    # assert result is True
    
    # TODO: Verify status was updated
    # server = manager.get_server("web-01")
    # assert server["status"] == "maintenance"
    pass

def test_update_nonexistent_server_status():
    """Test updating status of nonexistent server."""
    manager = ServerManager()
    
    # TODO: Try to update nonexistent server
    # result = manager.update_server_status("nonexistent", "active")
    # assert result is False
    pass

def test_get_servers_by_status():
    """Test getting servers by status."""
    manager = ServerManager()
    
    # TODO: Add servers with different statuses
    # manager.add_server("web-01", "192.168.1.10")
    # manager.add_server("web-02", "192.168.1.11")
    # manager.add_server("db-01", "192.168.1.20")
    
    # TODO: Update some statuses
    # manager.update_server_status("web-01", "maintenance")
    # manager.update_server_status("db-01", "stopped")
    
    # TODO: Test filtering by status
    # active_servers = manager.get_servers_by_status("active")
    # maintenance_servers = manager.get_servers_by_status("maintenance")
    # stopped_servers = manager.get_servers_by_status("stopped")
    
    # TODO: Verify results
    # assert "web-02" in active_servers
    # assert "web-01" in maintenance_servers
    # assert "db-01" in stopped_servers
    pass

def test_multiple_servers():
    """Test managing multiple servers."""
    manager = ServerManager()
    
    # TODO: Add multiple servers
    servers = [
        ("web-01", "192.168.1.10", 80),
        ("web-02", "192.168.1.11", 80),
        ("db-01", "192.168.1.20", 5432)
    ]
    
    # TODO: Add all servers and verify
    # for name, ip, port in servers:
    #     result = manager.add_server(name, ip, port)
    #     assert result is True
    
    # TODO: Check total count
    # assert manager.count_servers() == 3
    
    # TODO: Verify all servers exist in list
    # server_names = manager.list_servers()
    # assert "web-01" in server_names
    # assert "web-02" in server_names
    # assert "db-01" in server_names
    
    # TODO: Check specific server details
    # db_server = manager.get_server("db-01")
    # assert db_server["port"] == 5432
    pass

def test_health_checks():
    """Test health check functionality."""
    manager = ServerManager()
    
    # TODO: Add server first
    # manager.add_server("web-01", "192.168.1.10")
    
    # TODO: Add health check
    # result = manager.add_health_check("web-01", True)
    # assert result is True
    
    # TODO: Try health check for nonexistent server
    # result = manager.add_health_check("nonexistent", False)
    # assert result is False
    pass

# TODO: Write a fixture-based test
def test_with_populated_manager():
    """Test using a pre-populated manager."""
    # TODO: Create and populate a manager
    # manager = ServerManager()
    # Sample servers to add:
    # - web-01: 192.168.1.10:80
    # - web-02: 192.168.1.11:80  
    # - db-01: 192.168.1.20:5432
    
    # TODO: Test operations on the populated manager
    pass

def main():
    """
    Exercise Instructions:
    
    1. Complete the ServerManager class methods (remove the pass statements)
    
    2. Complete all the test functions (remove the pass statements and uncomment/complete the test code)
    
    3. Run the tests:
       pytest exercise2_server_testing.py -v
    
    4. Make sure all tests pass
    
    Tasks to complete:
    1. Implement ServerManager.remove_server()
    2. Implement ServerManager.get_server()
    3. Implement ServerManager.list_servers()
    4. Implement ServerManager.count_servers()
    5. Implement ServerManager.update_server_status()
    6. Implement ServerManager.get_servers_by_status()
    7. Implement ServerManager.add_health_check()
    8. Complete all test functions
    9. Add validation to add_server() method
    
    Bonus challenges:
    1. Add IP address validation
    2. Add port range validation (1-65535)
    3. Add tests for edge cases
    4. Create pytest fixtures for common test data
    5. Add parametrized tests for different server configurations
    """
    print("Exercise 7A.2: Testing a Server Manager")
    print("Complete the ServerManager class and all test functions!")
    print("Run with: pytest exercise2_server_testing.py -v")

if __name__ == "__main__":
    main()