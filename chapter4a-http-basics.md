# Chapter 4A: HTTP Basics for DevOps

## Learning Objectives
- Understand HTTP methods and status codes
- Make your first API requests
- Handle responses and errors
- Practice with simple tools

## 4A.1 What is HTTP?

HTTP (HyperText Transfer Protocol) is how computers talk to each other on the web. Think of it like sending letters:

- **Request**: You ask for something
- **Response**: The server sends back an answer

### HTTP Methods (Verbs)
- **GET**: "Give me information" (like reading a file)
- **POST**: "Create something new" (like adding a server)
- **PUT**: "Update/replace something" (like changing server config)
- **DELETE**: "Remove something" (like deleting a server)

### HTTP Status Codes
- **200 OK**: "Everything worked!"
- **404 Not Found**: "That doesn't exist"
- **500 Server Error**: "Something broke on their end"

## 4A.2 Your First API Request

Let's start simple with the `requests` library:

```python
# first_request.py
import requests

# Make your first API request
response = requests.get('https://httpbin.org/get')

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
```

### Exercise 4A.1: Make Your First Request
1. Create `first_request.py` with the code above
2. Run it: `python first_request.py`
3. Try changing the URL to: `https://httpbin.org/status/404`
4. What status code do you get?

**Expected Output:**
```
Status Code: 200
Status Code: 404
```

## 4A.3 Understanding JSON Responses

Most APIs return data in JSON format:

```python
# json_basics.py
import requests
import json

# Get JSON data
response = requests.get('https://httpbin.org/json')

if response.status_code == 200:
    data = response.json()  # Parse JSON automatically
    print("JSON Data:")
    print(json.dumps(data, indent=2))  # Pretty print
else:
    print(f"Error: {response.status_code}")
```

### Exercise 4A.2: Working with JSON
1. Create `json_basics.py` and run it
2. Print just one field from the JSON response
3. Try the URL: `https://api.github.com/users/octocat`
4. Print the user's name and public repositories count

**Solution hint:**
```python
# After getting the response
user_data = response.json()
print(f"Name: {user_data['name']}")
print(f"Public Repos: {user_data['public_repos']}")
```

## 4A.4 Sending Data with POST

POST requests send data to create something new:

```python
# post_example.py
import requests

# Data to send
data = {
    "name": "web-server-01",
    "ip": "192.168.1.10",
    "environment": "production"
}

# Send POST request
response = requests.post(
    'https://httpbin.org/post',
    json=data  # Sends as JSON
)

print(f"Status: {response.status_code}")
print("Server received our data:")
print(response.json()['json'])  # Echo back what we sent
```

### Exercise 4A.3: Send Server Data
1. Create `post_example.py` and run it
2. Modify the data to include a `port` field
3. Try sending the data as form data instead of JSON:
   ```python
   response = requests.post(url, data=data)  # Form data
   ```

## 4A.5 Adding Headers

Headers provide extra information about your request:

```python
# headers_example.py
import requests

# Custom headers
headers = {
    'User-Agent': 'DevOps-Tool/1.0',
    'Accept': 'application/json',
    'X-API-Key': 'your-api-key-here'
}

response = requests.get(
    'https://httpbin.org/headers',
    headers=headers
)

print("Headers we sent:")
received_headers = response.json()['headers']
for key, value in received_headers.items():
    print(f"  {key}: {value}")
```

### Exercise 4A.4: Custom Headers
1. Create `headers_example.py`
2. Add a custom header with your name
3. Try the GitHub API with a User-Agent header:
   ```python
   headers = {'User-Agent': 'Your-Name/1.0'}
   response = requests.get('https://api.github.com/users/octocat', headers=headers)
   ```

## 4A.6 Simple Error Handling

Always handle errors when making requests:

```python
# error_handling.py
import requests

def safe_api_request(url):
    """Make a safe API request with error handling."""
    try:
        response = requests.get(url, timeout=5)
        
        # Check if successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out!")
        return None
    except requests.exceptions.ConnectionError:
        print("Could not connect to the server!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test it
urls_to_test = [
    'https://httpbin.org/get',           # Should work
    'https://httpbin.org/status/500',    # Server error
    'https://non-existent-site.xyz',    # Connection error
]

for url in urls_to_test:
    print(f"Testing: {url}")
    result = safe_api_request(url)
    if result:
        print("  ✅ Success!")
    print()
```

### Exercise 4A.5: Error Handling Practice
1. Create `error_handling.py` and run it
2. Add a case for 404 errors specifically
3. Test with a URL that times out: `https://httpbin.org/delay/10`

## 4A.7 Mini Project: Server Status Checker

Put it all together with a simple tool:

```python
# server_checker.py
import requests
import json
from datetime import datetime

def check_server_status(url, name):
    """Check if a server is responding."""
    print(f"Checking {name}...")
    
    try:
        start_time = datetime.now()
        response = requests.get(url, timeout=5)
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        result = {
            'name': name,
            'url': url,
            'status_code': response.status_code,
            'response_time': round(response_time, 2),
            'is_healthy': response.status_code == 200,
            'checked_at': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {
            'name': name,
            'url': url,
            'status_code': None,
            'error': str(e),
            'is_healthy': False,
            'checked_at': datetime.now().isoformat()
        }

# Test multiple servers
servers = [
    ("GitHub API", "https://api.github.com"),
    ("HTTPBin", "https://httpbin.org/get"),
    ("Google", "https://www.google.com"),
]

print("Server Status Report")
print("=" * 30)

all_results = []
for name, url in servers:
    result = check_server_status(url, name)
    all_results.append(result)
    
    status = "✅ HEALTHY" if result['is_healthy'] else "❌ UNHEALTHY"
    print(f"{result['name']}: {status}")
    if 'response_time' in result:
        print(f"  Response time: {result['response_time']}s")
    if 'error' in result:
        print(f"  Error: {result['error']}")
    print()

# Save results to JSON file
with open('server_status.json', 'w') as f:
    json.dump(all_results, f, indent=2)
    
print("Results saved to server_status.json")
```

### Exercise 4A.6: Build Your Server Checker
1. Create `server_checker.py` and run it
2. Add your own servers to check
3. Modify it to check for specific status codes (like 200, 201)
4. Add a function to send results to a webhook (use https://httpbin.org/post)

## Key Takeaways

- **Start simple**: Begin with basic GET requests
- **Handle errors**: Always expect things to go wrong
- **Use JSON**: Most modern APIs use JSON format
- **Check status codes**: 200 means success, others indicate problems
- **Add timeouts**: Don't wait forever for responses

## Next Steps

In Chapter 4B, you'll learn about:
- Authentication with API keys
- Working with different response formats
- Building reusable API clients
- Monitoring and logging requests