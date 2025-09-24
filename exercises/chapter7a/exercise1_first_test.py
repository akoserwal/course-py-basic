#!/usr/bin/env python3
"""
Exercise 7A.1: Your First Unit Test
Learn to write basic unit tests with pytest.
"""

# Simple functions to test
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

def subtract_numbers(a, b):
    """Subtract b from a."""
    return a - b

def multiply_numbers(a, b):
    """Multiply two numbers."""
    return a * b

def divide_numbers(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def is_even(number):
    """Check if a number is even."""
    return number % 2 == 0

def get_server_url(name, port=80, protocol="http"):
    """Generate a server URL."""
    return f"{protocol}://{name}:{port}"

# TODO: Write tests for all the functions above
# Complete these test functions:

def test_add_numbers():
    """Test adding two numbers."""
    # TODO: Test that 2 + 3 = 5
    result = add_numbers(2, 3)
    assert result == 5
    
    # TODO: Test that 0 + 0 = 0
    # Add your test here...
    
    # TODO: Test with negative numbers: -5 + 3 = -2
    # Add your test here...

def test_subtract_numbers():
    """Test subtracting two numbers."""
    # TODO: Test that 5 - 3 = 2
    # Add your test here...
    
    # TODO: Test that 0 - 5 = -5
    # Add your test here...

def test_multiply_numbers():
    """Test multiplying two numbers."""
    # TODO: Test that 3 * 4 = 12
    # Add your test here...
    
    # TODO: Test that anything * 0 = 0
    # Add your test here...

def test_divide_numbers():
    """Test dividing two numbers."""
    # TODO: Test that 10 / 2 = 5
    # Add your test here...
    
    # TODO: Test that 7 / 2 = 3.5
    # Add your test here...

def test_divide_by_zero():
    """Test that dividing by zero raises an error."""
    # TODO: Use pytest.raises to test that dividing by zero raises ValueError
    # Hint: import pytest first, then use:
    # with pytest.raises(ValueError, match="Cannot divide by zero"):
    #     divide_numbers(10, 0)
    pass  # Remove this when you add your test

def test_is_even():
    """Test checking if numbers are even."""
    # TODO: Test that 2 is even (should return True)
    # Add your test here...
    
    # TODO: Test that 3 is not even (should return False)
    # Add your test here...
    
    # TODO: Test that 0 is even
    # Add your test here...

def test_get_server_url():
    """Test generating server URLs."""
    # TODO: Test basic URL generation
    # get_server_url("localhost") should return "http://localhost:80"
    # Add your test here...
    
    # TODO: Test with custom port
    # get_server_url("example.com", 8080) should return "http://example.com:8080"
    # Add your test here...
    
    # TODO: Test with custom protocol
    # get_server_url("secure.com", 443, "https") should return "https://secure.com:443"
    # Add your test here...

# TODO: Add your own function and test
# Create a function that checks if a string is a valid IP address (simple check)
# Then write tests for it

def is_valid_ip(ip_string):
    """Check if a string looks like a valid IP address (simple check)."""
    # TODO: Implement this function
    # Hint: Split by '.' and check if there are 4 parts, all numbers between 0-255
    pass

def test_is_valid_ip():
    """Test IP address validation."""
    # TODO: Test valid IPs like "192.168.1.1"
    # TODO: Test invalid IPs like "256.1.1.1" or "192.168.1"
    pass

def main():
    """
    Exercise Instructions:
    
    1. Install pytest if you haven't already:
       pip install pytest
    
    2. Complete all the TODO sections in this file
    
    3. Run the tests:
       pytest exercise1_first_test.py -v
    
    4. Make sure all tests pass
    
    5. Try making a test fail on purpose to see what happens
    
    6. Bonus challenges:
       a. Add tests for edge cases (very large numbers, negative numbers)
       b. Add a function to calculate factorial and test it
       c. Add a function to check if a year is a leap year and test it
       d. Use pytest's parametrize decorator to test multiple inputs
    
    Tips:
    - Use clear, descriptive test names
    - Test both normal cases and edge cases
    - Use assert statements to check results
    - Import pytest to use pytest.raises for exception testing
    """
    print("Exercise 7A.1: Your First Unit Test")
    print("Complete the TODO sections and run with: pytest exercise1_first_test.py -v")

if __name__ == "__main__":
    main()