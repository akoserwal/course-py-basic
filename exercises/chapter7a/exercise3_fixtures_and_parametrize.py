#!/usr/bin/env python3
"""
Exercise 7A.3: Fixtures and Parametrized Tests
Learn to use pytest fixtures and parametrized tests for better test organization.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Simple config manager to test
class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = {}
        if config_file and Path(config_file).exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip()
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: str):
        """Set configuration value."""
        self.config[key] = value
    
    def save_config(self):
        """Save configuration to file."""
        if not self.config_file:
            raise ValueError("No config file specified")
        
        with open(self.config_file, 'w') as f:
            for key, value in self.config.items():
                f.write(f"{key}={value}\n")
    
    def get_all_keys(self) -> List[str]:
        """Get all configuration keys."""
        return list(self.config.keys())
    
    def clear(self):
        """Clear all configuration."""
        self.config.clear()

# TODO: Create fixtures for common test scenarios

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    # TODO: Create a temporary file with sample configuration
    # Sample config content:
    config_content = """# Application Configuration
database_host=localhost
database_port=5432
database_name=testdb
api_timeout=30
debug_mode=true
# This is a comment
log_level=INFO
"""
    
    # TODO: Create temporary file
    # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
    #     f.write(config_content)
    #     temp_path = f.name
    
    # TODO: Yield the file path for testing
    # yield temp_path
    
    # TODO: Cleanup after test
    # Path(temp_path).unlink(missing_ok=True)
    pass

@pytest.fixture
def empty_config_manager():
    """Create an empty ConfigManager for testing."""
    # TODO: Return a new ConfigManager instance with no config file
    pass

@pytest.fixture
def populated_config_manager(temp_config_file):
    """Create a ConfigManager with sample data loaded."""
    # TODO: Create ConfigManager with the temp config file
    # return ConfigManager(temp_config_file)
    pass

@pytest.fixture
def config_with_different_data():
    """Create config manager with different test data."""
    # TODO: Create a ConfigManager and manually set some config values
    # manager = ConfigManager()
    # manager.set("server_name", "test-server")
    # manager.set("port", "8080")
    # manager.set("environment", "test")
    # return manager
    pass

# TODO: Write tests using fixtures

def test_empty_config_manager(empty_config_manager):
    """Test empty config manager using fixture."""
    # TODO: Test that empty manager has no keys
    # assert len(empty_config_manager.get_all_keys()) == 0
    
    # TODO: Test getting nonexistent key returns default
    # assert empty_config_manager.get("nonexistent", "default") == "default"
    pass

def test_populated_config_manager(populated_config_manager):
    """Test populated config manager using fixture."""
    # TODO: Test that config was loaded
    # assert len(populated_config_manager.get_all_keys()) > 0
    
    # TODO: Test specific values
    # assert populated_config_manager.get("database_host") == "localhost"
    # assert populated_config_manager.get("database_port") == "5432"
    # assert populated_config_manager.get("debug_mode") == "true"
    pass

def test_config_file_loading(temp_config_file):
    """Test loading config from file using fixture."""
    # TODO: Create manager with temp file
    # manager = ConfigManager(temp_config_file)
    
    # TODO: Test that values were loaded correctly
    # assert manager.get("database_host") == "localhost"
    # assert manager.get("api_timeout") == "30"
    # assert manager.get("log_level") == "INFO"
    pass

def test_config_modification(config_with_different_data):
    """Test modifying config using fixture."""
    # TODO: Test getting existing values
    # assert config_with_different_data.get("server_name") == "test-server"
    
    # TODO: Test setting new values
    # config_with_different_data.set("new_key", "new_value")
    # assert config_with_different_data.get("new_key") == "new_value"
    
    # TODO: Test overwriting existing values
    # config_with_different_data.set("port", "9090")
    # assert config_with_different_data.get("port") == "9090"
    pass

# TODO: Create parametrized tests for different config scenarios

# Test data for config key-value pairs
config_test_data = [
    ("database_host", "localhost"),
    ("database_port", "5432"), 
    ("database_name", "testdb"),
    ("api_timeout", "30"),
    ("debug_mode", "true"),
    ("log_level", "INFO"),
]

@pytest.mark.parametrize("key,expected_value", config_test_data)
def test_config_values_parametrized(populated_config_manager, key, expected_value):
    """Test config values using parametrized test."""
    # TODO: Test that each key has the expected value
    # assert populated_config_manager.get(key) == expected_value
    pass

# Test data for different config file formats
config_file_test_data = [
    ("key1=value1", "key1", "value1"),
    ("  key2  =  value2  ", "key2", "value2"),  # With spaces
    ("key3=value=with=equals", "key3", "value=with=equals"),  # Value with equals
    ("key4=", "key4", ""),  # Empty value
]

@pytest.mark.parametrize("line,expected_key,expected_value", config_file_test_data)
def test_config_line_parsing(line, expected_key, expected_value):
    """Test parsing different config line formats."""
    # TODO: Create temporary file with single line
    # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
    #     f.write(line)
    #     temp_path = f.name
    
    # TODO: Load config and test
    # try:
    #     manager = ConfigManager(temp_path)
    #     assert manager.get(expected_key) == expected_value
    # finally:
    #     Path(temp_path).unlink(missing_ok=True)
    pass

# Test data for invalid config scenarios
invalid_config_data = [
    ("invalid_line_without_equals", False),
    ("# comment_line_only", True),  # Should be ignored
    ("", True),  # Empty line should be ignored
    ("   ", True),  # Whitespace only should be ignored
]

@pytest.mark.parametrize("line,should_be_ignored", invalid_config_data)
def test_invalid_config_lines(line, should_be_ignored):
    """Test handling of invalid or special config lines."""
    # TODO: Create temporary file with the test line
    # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
    #     f.write(line)
    #     temp_path = f.name
    
    # TODO: Test loading the config
    # try:
    #     manager = ConfigManager(temp_path)
    #     if should_be_ignored:
    #         # Line should be ignored, no keys should be loaded
    #         assert len(manager.get_all_keys()) == 0
    #     else:
    #         # Should handle gracefully or raise appropriate error
    #         pass  # Implement based on expected behavior
    # finally:
    #     Path(temp_path).unlink(missing_ok=True)
    pass

# Test data for get() method with defaults
get_default_test_data = [
    ("existing_key", "default_value", "actual_value"),  # Existing key should return actual value
    ("nonexistent_key", "default_value", "default_value"),  # Non-existing should return default
    ("another_key", None, None),  # Default None should work
    ("key_with_empty", "", ""),  # Empty string default
]

@pytest.mark.parametrize("key,default,expected", get_default_test_data)
def test_get_with_defaults_parametrized(empty_config_manager, key, default, expected):
    """Test get() method with different default values."""
    # TODO: Set up test data if needed
    # if expected == "actual_value":
    #     empty_config_manager.set(key, "actual_value")
    
    # TODO: Test get with default
    # result = empty_config_manager.get(key, default)
    # assert result == expected
    pass

# TODO: Create tests for edge cases and error conditions

def test_load_nonexistent_file():
    """Test loading a config file that doesn't exist."""
    # TODO: Test creating ConfigManager with nonexistent file
    # Should create empty config, not raise error
    # manager = ConfigManager("/nonexistent/file.conf")
    # assert len(manager.get_all_keys()) == 0
    pass

def test_save_config_without_file():
    """Test saving config when no file was specified."""
    # TODO: Test that saving without config_file raises error
    # manager = ConfigManager()
    # manager.set("key", "value")
    
    # with pytest.raises(ValueError, match="No config file specified"):
    #     manager.save_config()
    pass

def test_save_and_reload_config():
    """Test saving config and reloading it."""
    # TODO: Create temp file, save config, reload and verify
    # with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
    #     temp_path = f.name
    
    # try:
    #     # Create manager and set values
    #     manager = ConfigManager(temp_path)
    #     manager.set("test_key", "test_value")
    #     manager.set("another_key", "another_value")
    
    #     # Save config
    #     manager.save_config()
    
    #     # Create new manager and load same file
    #     new_manager = ConfigManager(temp_path)
    
    #     # Verify values were saved and loaded correctly
    #     assert new_manager.get("test_key") == "test_value"
    #     assert new_manager.get("another_key") == "another_value"
    
    # finally:
    #     Path(temp_path).unlink(missing_ok=True)
    pass

def test_clear_config(populated_config_manager):
    """Test clearing all configuration."""
    # TODO: Verify config has data initially
    # assert len(populated_config_manager.get_all_keys()) > 0
    
    # TODO: Clear config
    # populated_config_manager.clear()
    
    # TODO: Verify config is empty
    # assert len(populated_config_manager.get_all_keys()) == 0
    # assert populated_config_manager.get("database_host") is None
    pass

def main():
    """
    Exercise Instructions:
    
    1. Complete all the fixture functions (remove pass statements)
    
    2. Complete all the test functions (uncomment and complete the test code)
    
    3. Run the tests:
       pytest exercise3_fixtures_and_parametrize.py -v
    
    4. Run with coverage:
       pytest exercise3_fixtures_and_parametrize.py --cov=exercise3_fixtures_and_parametrize
    
    Tasks to complete:
    1. Implement all fixture functions
    2. Complete all test functions using the fixtures
    3. Implement parametrized tests
    4. Add edge case testing
    5. Test error conditions
    
    Key concepts to practice:
    - Creating and using pytest fixtures
    - Parametrized testing with @pytest.mark.parametrize
    - Temporary file handling in tests
    - Testing both happy path and error conditions
    - Using fixtures for test data setup
    
    Bonus challenges:
    1. Add fixture scope testing (function, class, module)
    2. Create fixtures that depend on other fixtures
    3. Add more parametrized test scenarios
    4. Test configuration inheritance or layering
    5. Add performance testing for large config files
    """
    print("Exercise 7A.3: Fixtures and Parametrized Tests")
    print("Learn to use pytest fixtures and parametrized tests!")
    print("Run with: pytest exercise3_fixtures_and_parametrize.py -v")

if __name__ == "__main__":
    main()