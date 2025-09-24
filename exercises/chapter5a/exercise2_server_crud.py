#!/usr/bin/env python3
"""
Exercise 5A.2: Server CRUD Operations
Learn to Create, Read, Update, and Delete servers using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Dict, List, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="Server CRUD API",
    description="Learn CRUD operations with servers",
    version="1.0.0"
)

# Data models
class ServerCreate(BaseModel):
    """Model for creating a new server."""
    name: str
    ip: str
    port: int = 80
    environment: str = "production"
    
    # TODO: Add validation
    # Uncomment and complete these validators:
    #
    # @validator('name')
    # def validate_name(cls, v):
    #     if len(v) < 3:
    #         raise ValueError('Name must be at least 3 characters')
    #     return v
    #
    # @validator('ip')
    # def validate_ip(cls, v):
    #     # Simple IP validation
    #     parts = v.split('.')
    #     if len(parts) != 4:
    #         raise ValueError('Invalid IP format')
    #     return v

class ServerResponse(BaseModel):
    """Model for server responses."""
    id: str
    name: str
    ip: str
    port: int
    environment: str
    status: str
    created_at: datetime

# In-memory storage (in real apps, use a database!)
servers: Dict[str, dict] = {}

@app.get("/")
def root():
    """Welcome message."""
    return {
        "message": "Server CRUD API",
        "endpoints": {
            "create_server": "POST /servers",
            "list_servers": "GET /servers", 
            "get_server": "GET /servers/{server_id}",
            "update_server": "PUT /servers/{server_id}",
            "delete_server": "DELETE /servers/{server_id}"
        },
        "total_servers": len(servers)
    }

# CREATE - Add a new server
@app.post("/servers", response_model=ServerResponse)
def create_server(server: ServerCreate):
    """Create a new server."""
    
    # TODO: Check for duplicate names
    # Uncomment and complete this check:
    #
    # for existing_server in servers.values():
    #     if existing_server["name"] == server.name:
    #         raise HTTPException(
    #             status_code=400,
    #             detail=f"Server with name '{server.name}' already exists"
    #         )
    
    # Generate a unique ID
    server_id = str(uuid.uuid4())[:8]  # Short UUID
    
    # Create server data
    server_data = {
        "id": server_id,
        "name": server.name,
        "ip": server.ip,
        "port": server.port,
        "environment": server.environment,
        "status": "created",
        "created_at": datetime.now()
    }
    
    # Store the server
    servers[server_id] = server_data
    
    return ServerResponse(**server_data)

# READ - Get all servers
@app.get("/servers", response_model=List[ServerResponse])
def list_servers():
    """Get a list of all servers."""
    # TODO: Convert stored servers to response models
    # Complete this function:
    
    server_list = []
    for server_data in servers.values():
        server_list.append(ServerResponse(**server_data))
    return server_list

# READ - Get a specific server
@app.get("/servers/{server_id}", response_model=ServerResponse)
def get_server(server_id: str):
    """Get a specific server by ID."""
    
    # TODO: Check if server exists and return it
    # Complete this function:
    
    if server_id not in servers:
        raise HTTPException(
            status_code=404,
            detail=f"Server with ID '{server_id}' not found"
        )
    
    return ServerResponse(**servers[server_id])

# UPDATE - Update server status
@app.put("/servers/{server_id}/status")
def update_server_status(server_id: str, new_status: str):
    """Update the status of a server."""
    
    # TODO: Implement server status update
    # Valid statuses: created, running, stopped, maintenance
    
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    valid_statuses = ["created", "running", "stopped", "maintenance"]
    
    # TODO: Validate the new status
    # Add validation here...
    
    # TODO: Update the server status
    # Add update logic here...
    
    return {"message": f"Server {server_id} status updated to {new_status}"}

# DELETE - Remove a server
@app.delete("/servers/{server_id}")
def delete_server(server_id: str):
    """Delete a server."""
    
    # TODO: Implement server deletion
    # Complete this function:
    
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    deleted_server = servers.pop(server_id)
    
    return {
        "message": "Server deleted successfully",
        "deleted_server": deleted_server["name"]
    }

# BONUS: Statistics endpoint
@app.get("/stats")
def get_stats():
    """Get server statistics."""
    if not servers:
        return {"total": 0, "by_status": {}, "by_environment": {}}
    
    # TODO: Calculate statistics
    # Count servers by status and environment
    
    status_counts = {}
    env_counts = {}
    
    # TODO: Complete the counting logic
    # for server in servers.values():
    #     status = server["status"]
    #     env = server["environment"]
    #     # Add counting logic here...
    
    return {
        "total": len(servers),
        "by_status": status_counts,
        "by_environment": env_counts
    }

# Helper function for testing
def create_sample_servers():
    """Create some sample servers for testing."""
    sample_servers = [
        ServerCreate(name="web-01", ip="192.168.1.10", port=80, environment="production"),
        ServerCreate(name="web-02", ip="192.168.1.11", port=80, environment="production"),
        ServerCreate(name="db-01", ip="192.168.1.20", port=5432, environment="production"),
        ServerCreate(name="test-server", ip="192.168.1.100", port=8080, environment="development"),
    ]
    
    for server_create in sample_servers:
        create_server(server_create)

def main():
    """
    Exercise Instructions:
    
    1. Complete the TODO sections in this file
    
    2. Run the server:
       uvicorn exercise2_server_crud:app --reload
    
    3. Visit http://localhost:8000/docs to test your API
    
    4. Try these operations in order:
       a. Create a server (POST /servers)
       b. List all servers (GET /servers)
       c. Get a specific server (GET /servers/{id})
       d. Update server status (PUT /servers/{id}/status)
       e. Delete a server (DELETE /servers/{id})
       f. Check statistics (GET /stats)
    
    5. Test error cases:
       - Try to get a server that doesn't exist
       - Try to create a server with invalid data
       - Try to delete a server that doesn't exist
    
    Exercises to complete:
    1. Add validation to ServerCreate model
    2. Implement duplicate name checking in create_server
    3. Complete the list_servers function
    4. Implement update_server_status with validation
    5. Complete the statistics calculation
    6. Add error handling throughout
    
    Bonus challenges:
    1. Add filtering to list_servers (by environment, status)
    2. Add a batch delete endpoint
    3. Add server health check endpoint
    4. Add sorting to server lists
    """
    print("FastAPI Exercise 2: Server CRUD Operations")
    print("Complete the TODO sections and test your API!")

if __name__ == "__main__":
    main()