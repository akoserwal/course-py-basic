#!/usr/bin/env python3
"""
Exercise 4A.3: Simple Server Status Checker
Build a tool to check if servers are responding.
"""

import requests
import json
from datetime import datetime

def check_single_server(url, name="Unknown"):
    """
    Check if a single server is responding.
    
    Args:
        url (str): The URL to check
        name (str): Friendly name for the server
    
    Returns:
        dict: Status information
    """
    print(f"Checking {name}...")
    
    try:
        # TODO: Make a request with a 5-second timeout
        response = requests.get(url, timeout=5)
        
        # TODO: Determine if the server is healthy
        is_healthy = response.status_code == 200
        
        # TODO: Create a result dictionary with:
        # - name, url, status_code, is_healthy, checked_at
        result = {
            'name': name,
            'url': url,
            'status_code': response.status_code,
            'is_healthy': is_healthy,
            'checked_at': datetime.now().isoformat()
        }
        
        return result
        
    except requests.exceptions.Timeout:
        # TODO: Handle timeout errors
        return {
            'name': name,
            'url': url,
            'error': 'Request timed out',
            'is_healthy': False,
            'checked_at': datetime.now().isoformat()
        }
    
    except requests.exceptions.ConnectionError:
        # TODO: Handle connection errors
        return {
            'name': name,
            'url': url,
            'error': 'Could not connect',
            'is_healthy': False,
            'checked_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        # TODO: Handle any other errors
        return {
            'name': name,
            'url': url,
            'error': str(e),
            'is_healthy': False,
            'checked_at': datetime.now().isoformat()
        }

def print_status_report(results):
    """Print a formatted status report."""
    print("\n" + "=" * 50)
    print("SERVER STATUS REPORT")
    print("=" * 50)
    
    healthy_count = 0
    total_count = len(results)
    
    for result in results:
        status = "✅ HEALTHY" if result['is_healthy'] else "❌ UNHEALTHY"
        print(f"\n{result['name']}: {status}")
        print(f"  URL: {result['url']}")
        
        if 'status_code' in result:
            print(f"  Status Code: {result['status_code']}")
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        
        if result['is_healthy']:
            healthy_count += 1
    
    print(f"\nSummary: {healthy_count}/{total_count} servers healthy")
    print(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def save_results_to_file(results, filename="server_status.json"):
    """Save results to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    """Main function to run the server checker."""
    print("Exercise 4A.3: Simple Server Status Checker")
    print("=" * 45)
    
    # TODO: Define servers to check
    # Add more servers to this list!
    servers_to_check = [
        ("GitHub API", "https://api.github.com"),
        ("HTTPBin", "https://httpbin.org/get"),
        ("Google", "https://www.google.com"),
        ("Fake Server", "https://this-does-not-exist-12345.com"),
    ]
    
    # TODO: Check all servers
    all_results = []
    
    for name, url in servers_to_check:
        result = check_single_server(url, name)
        all_results.append(result)
    
    # TODO: Print the report
    print_status_report(all_results)
    
    # TODO: Save results to file
    save_results_to_file(all_results)
    
    # Challenge tasks:
    print("\n" + "=" * 50)
    print("CHALLENGE TASKS:")
    print("1. Add more servers to check")
    print("2. Add response time measurement")
    print("3. Check for specific content in the response")
    print("4. Send results to a webhook (use httpbin.org/post)")
    print("5. Add retry logic for failed requests")

def challenge_response_time():
    """Challenge: Add response time measurement."""
    print("\nChallenge: Measuring Response Time")
    print("-" * 35)
    
    url = "https://httpbin.org/get"
    
    # TODO: Measure how long the request takes
    start_time = datetime.now()
    response = requests.get(url)
    end_time = datetime.now()
    
    response_time = (end_time - start_time).total_seconds()
    
    print(f"URL: {url}")
    print(f"Status: {response.status_code}")
    print(f"Response Time: {response_time:.3f} seconds")

def challenge_content_check():
    """Challenge: Check for specific content."""
    print("\nChallenge: Content Verification")
    print("-" * 32)
    
    # TODO: Check if the response contains expected content
    url = "https://httpbin.org/get"
    response = requests.get(url)
    
    # Check if response contains certain text
    expected_content = "httpbin"
    has_expected_content = expected_content in response.text.lower()
    
    print(f"URL: {url}")
    print(f"Contains '{expected_content}': {has_expected_content}")

if __name__ == "__main__":
    main()
    
    # Uncomment these to try the challenges:
    # challenge_response_time()
    # challenge_content_check()