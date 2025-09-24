#!/usr/bin/env python3
"""
Exercise 5A.1: Your First FastAPI Application
Learn to create basic API endpoints and understand the fundamentals.
"""

from fastapi import FastAPI
from datetime import datetime

# Create the FastAPI application
app = FastAPI(
    title="My First API",
    description="Learning FastAPI basics",
    version="1.0.0"
)

# Basic endpoint
@app.get("/")
def read_root():
    """The root endpoint - like the front door of your API."""
    return {"message": "Hello, DevOps World!"}

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint - is the server alive?"""
    return {
        "status": "healthy", 
        "service": "my-first-api",
        "timestamp": datetime.now().isoformat()
    }

# TODO: Add your own endpoints here!

# Exercise 1: Add an /info endpoint
# Uncomment and complete this function:
#
# @app.get("/info")
# def get_info():
#     """Information about the API and developer."""
#     return {
#         "developer": "Your Name Here",
#         "current_time": datetime.now().isoformat(),
#         "version": "1.0.0",
#         "description": "My first FastAPI application"
#     }

# Exercise 2: Add a /greet/{name} endpoint
# This should greet a person by name
# Example: /greet/Alice should return {"message": "Hello, Alice!"}
#
# TODO: Create this endpoint

# Exercise 3: Add a /math/add/{a}/{b} endpoint
# This should add two numbers and return the result
# Example: /math/add/5/3 should return {"a": 5, "b": 3, "result": 8}
#
# TODO: Create this endpoint

# Exercise 4: Add a /server-time endpoint with timezone info
# Use the datetime module to show different time formats
#
# TODO: Create this endpoint

def main():
    """
    Instructions for running this exercise:
    
    1. Install FastAPI and uvicorn:
       pip install fastapi uvicorn
    
    2. Save this file as exercise1_hello_api.py
    
    3. Run the server:
       uvicorn exercise1_hello_api:app --reload
    
    4. Visit these URLs in your browser:
       - http://localhost:8000/ (root endpoint)
       - http://localhost:8000/health (health check)
       - http://localhost:8000/docs (automatic documentation!)
    
    5. Complete the TODO exercises above
    
    6. Check your work by visiting the /docs page and testing each endpoint
    
    Tips:
    - The --reload flag restarts the server when you change the code
    - The /docs page shows interactive documentation
    - You can test endpoints directly from the /docs page
    """
    print("FastAPI Exercise 1: Your First API")
    print("Follow the instructions in the main() function!")

if __name__ == "__main__":
    main()