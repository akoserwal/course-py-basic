#!/usr/bin/env python3
"""
Exercise 4A.2: Working with JSON Responses
Learn to parse and work with JSON data from APIs.
"""

import requests
import json

def demo_json_parsing():
    """Demonstrate basic JSON parsing."""
    print("Demo: Basic JSON parsing")
    print("-" * 30)
    
    response = requests.get('https://httpbin.org/json')
    
    if response.status_code == 200:
        data = response.json()  # Parse JSON
        print("Full JSON response:")
        print(json.dumps(data, indent=2))
        print()
        
        # Access specific fields
        print("Accessing specific fields:")
        try:
            slideshow = data['slideshow']
            print(f"Title: {slideshow['title']}")
            print(f"Author: {slideshow['author']}")
        except KeyError as e:
            print(f"Field not found: {e}")
    print()

def github_user_exercise():
    """Exercise with GitHub API."""
    print("Exercise: GitHub User Information")
    print("-" * 35)
    
    # TODO: Make a request to GitHub API for user 'octocat'
    # URL: https://api.github.com/users/octocat
    
    print("Task 1: Get GitHub user 'octocat' information")
    
    # TODO: Uncomment and complete this code:
    # url = "https://api.github.com/users/octocat"
    # response = requests.get(url)
    # 
    # if response.status_code == 200:
    #     user_data = response.json()
    #     
    #     # TODO: Print the following information:
    #     # - User's real name
    #     # - Number of public repositories
    #     # - Number of followers
    #     # - Account creation date
    #     
    #     print(f"Name: {user_data['name']}")
    #     print(f"Public Repos: {user_data['public_repos']}")
    #     # Add more fields here...
    # else:
    #     print(f"Error: {response.status_code}")
    
    print("(Uncomment the code above to complete this exercise)")
    print()

def weather_api_exercise():
    """Exercise with a weather API (demo data)."""
    print("Exercise: Weather Data Processing")
    print("-" * 33)
    
    # Using httpbin to simulate weather data
    weather_data = {
        "location": "New York",
        "temperature": 72,
        "humidity": 65,
        "conditions": "Partly Cloudy",
        "forecast": [
            {"day": "Today", "high": 75, "low": 60},
            {"day": "Tomorrow", "high": 78, "low": 62},
            {"day": "Wednesday", "high": 73, "low": 59}
        ]
    }
    
    print("Weather data (simulated):")
    print(json.dumps(weather_data, indent=2))
    print()
    
    # TODO: Extract and display the following:
    print("Tasks:")
    print("1. Current temperature and conditions")
    print("2. Today's high and low temperatures")
    print("3. Average high temperature for the forecast")
    print()
    
    # TODO: Complete these tasks:
    print("Solutions:")
    print(f"1. Current: {weather_data['temperature']}°F, {weather_data['conditions']}")
    
    # TODO: Find today's forecast and print high/low
    today_forecast = None
    for day in weather_data['forecast']:
        if day['day'] == 'Today':
            today_forecast = day
            break
    
    if today_forecast:
        print(f"2. Today: High {today_forecast['high']}°F, Low {today_forecast['low']}°F")
    
    # TODO: Calculate average high temperature
    # Add your code here...
    print()

def main():
    """Run all exercises."""
    print("Exercise 4A.2: Working with JSON")
    print("=" * 40)
    print()
    
    demo_json_parsing()
    github_user_exercise()
    weather_api_exercise()
    
    print("Challenge: Try these APIs on your own:")
    print("- https://api.github.com/repos/python/cpython")
    print("- https://httpbin.org/uuid")
    print("- https://httpbin.org/ip")

if __name__ == "__main__":
    main()