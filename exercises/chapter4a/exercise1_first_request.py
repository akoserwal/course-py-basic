#!/usr/bin/env python3
"""
Exercise 4A.1: Your First API Request
Learn to make basic HTTP requests and understand status codes.
"""

import requests

def main():
    print("Exercise 4A.1: Making Your First API Request")
    print("=" * 50)
    
    # Task 1: Make a successful request
    print("Task 1: Making a successful request...")
    response = requests.get('https://httpbin.org/get')
    print(f"Status Code: {response.status_code}")
    print(f"Success: {response.status_code == 200}")
    print()
    
    # Task 2: Try a 404 error
    print("Task 2: Testing a 404 error...")
    response = requests.get('https://httpbin.org/status/404')
    print(f"Status Code: {response.status_code}")
    print(f"Is this a 404? {response.status_code == 404}")
    print()
    
    # Task 3: Your turn! 
    print("Task 3: Your turn to try different status codes:")
    print("Try these URLs and predict the status code:")
    
    test_urls = [
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/401', 
        'https://httpbin.org/status/500',
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        # TODO: Make a request to this URL
        # TODO: Print the status code
        # TODO: Print what this status code means
        
        # Solution (uncomment to see):
        # response = requests.get(url)
        # print(f"Status Code: {response.status_code}")
        
        # Status code meanings:
        # 200 = OK (success)
        # 401 = Unauthorized (need authentication)
        # 500 = Internal Server Error (server problem)

if __name__ == "__main__":
    main()