# Chapter 5A: FastAPI Basics - Your First API

## Learning Objectives
- Create your first FastAPI application
- Understand routes and HTTP methods
- Handle request data and responses
- Test your API with simple tools

## 5A.1 Why FastAPI?

FastAPI is a modern Python web framework for building APIs quickly and easily:
- **Fast to code**: Write APIs in minutes, not hours
- **Fast execution**: High performance 
- **Automatic docs**: Interactive API documentation
- **Type hints**: Uses Python type hints for validation

Think of it as creating a restaurant menu - you define what's available, and customers can order from it!

## 5A.2 Your First API

Let's start with the simplest possible API:

```python
# hello_api.py
from fastapi import FastAPI

# Create the API application
app = FastAPI()

@app.get("/")
def read_root():
    """The root endpoint - like the front door of your API."""
    return {"message": "Hello, DevOps World!"}

@app.get("/health")
def health_check():
    """Health check endpoint - is the server alive?"""
    return {"status": "healthy", "service": "my-first-api"}
```

### Running Your API
```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn

# Run your API
uvicorn hello_api:app --reload
```

Visit: http://localhost:8000
- Your API: http://localhost:8000/
- Health check: http://localhost:8000/health
- Auto docs: http://localhost:8000/docs (This is amazing!)

### Exercise 5A.1: Your First API
1. Create `hello_api.py` with the code above
2. Run it and visit all the URLs
3. Add a new endpoint `/info` that returns your name and the current time
4. Look at the automatic documentation at `/docs`

**Solution hint:**
```python
from datetime import datetime

@app.get("/info")
def get_info():
    return {
        "developer": "Your Name",
        "current_time": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

## 5A.3 Understanding HTTP Methods

Different HTTP methods do different things:

```python
# methods_demo.py
from fastapi import FastAPI

app = FastAPI()

# GET - Read data
@app.get("/servers")
def list_servers():
    """Get a list of all servers."""
    return {"servers": ["web-01", "web-02", "db-01"]}

# POST - Create new data
@app.post("/servers")
def create_server():
    """Create a new server."""
    return {"message": "Server created", "id": "web-03"}

# PUT - Update data
@app.put("/servers/{server_id}")
def update_server(server_id: str):
    """Update a specific server."""
    return {"message": f"Server {server_id} updated"}

# DELETE - Remove data
@app.delete("/servers/{server_id}")
def delete_server(server_id: str):
    """Delete a specific server."""
    return {"message": f"Server {server_id} deleted"}
```

### Exercise 5A.2: HTTP Methods
1. Create `methods_demo.py` and run it
2. Visit `/docs` and try each method using the interactive interface
3. Notice how URLs with `{server_id}` create parameters
4. Add a GET endpoint to get a single server: `/servers/{server_id}`

## 5A.4 Handling Data Input

APIs need to receive data from users:

```python
# data_input.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Define what data we expect
class ServerCreate(BaseModel):
    name: str
    ip: str
    port: int = 80  # Default value
    environment: Optional[str] = "production"

# Store servers in memory (simple for now)
servers = {}

@app.post("/servers")
def create_server(server: ServerCreate):
    """Create a new server with the provided data."""
    server_id = f"server_{len(servers) + 1}"
    
    # Convert to dictionary and add ID
    server_data = server.dict()
    server_data["id"] = server_id
    
    # Store it
    servers[server_id] = server_data
    
    return {"message": "Server created", "server": server_data}

@app.get("/servers")
def list_servers():
    """Get all servers."""
    return {"servers": list(servers.values())}

@app.get("/servers/{server_id}")
def get_server(server_id: str):
    """Get a specific server."""
    if server_id not in servers:
        return {"error": "Server not found"}, 404
    
    return {"server": servers[server_id]}
```

### Exercise 5A.3: Data Input
1. Create `data_input.py` and run it
2. Use the `/docs` interface to create servers with different data
3. List the servers to see what was created
4. Try creating a server with missing data - what happens?
5. Add validation: server names must be at least 3 characters

**Validation hint:**
```python
from pydantic import BaseModel, validator

class ServerCreate(BaseModel):
    name: str
    ip: str
    port: int = 80
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if len(v) < 3:
            raise ValueError('Name must be at least 3 characters')
        return v
```

## 5A.5 Error Handling

Handle errors gracefully:

```python
# error_handling.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ServerCreate(BaseModel):
    name: str
    ip: str

servers = {}

@app.post("/servers")
def create_server(server: ServerCreate):
    """Create a server with error checking."""
    
    # Check if server already exists
    for existing_server in servers.values():
        if existing_server["name"] == server.name:
            raise HTTPException(
                status_code=400, 
                detail=f"Server with name '{server.name}' already exists"
            )
    
    # Validate IP format (simple check)
    ip_parts = server.ip.split(".")
    if len(ip_parts) != 4:
        raise HTTPException(
            status_code=400,
            detail="Invalid IP address format"
        )
    
    # Create the server
    server_id = f"server_{len(servers) + 1}"
    server_data = server.dict()
    server_data["id"] = server_id
    
    servers[server_id] = server_data
    
    return {"message": "Server created successfully", "server": server_data}

@app.get("/servers/{server_id}")
def get_server(server_id: str):
    """Get a server with proper error handling."""
    if server_id not in servers:
        raise HTTPException(
            status_code=404,
            detail=f"Server with ID '{server_id}' not found"
        )
    
    return {"server": servers[server_id]}

@app.delete("/servers/{server_id}")
def delete_server(server_id: str):
    """Delete a server."""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    deleted_server = servers.pop(server_id)
    return {"message": "Server deleted", "server": deleted_server}
```

### Exercise 5A.4: Error Handling
1. Create `error_handling.py` and run it
2. Try to create duplicate servers - you should get an error
3. Try to get a server that doesn't exist
4. Add error handling for invalid port numbers (must be 1-65535)

## 5A.6 Path and Query Parameters

Get data from the URL:

```python
# parameters.py
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

# Dummy data
servers = {
    "server_1": {"name": "web-01", "environment": "prod", "status": "running"},
    "server_2": {"name": "web-02", "environment": "prod", "status": "stopped"},
    "server_3": {"name": "db-01", "environment": "dev", "status": "running"},
}

@app.get("/servers/{server_id}")
def get_server(server_id: str):
    """Path parameter: server_id comes from the URL."""
    if server_id not in servers:
        return {"error": "Server not found"}, 404
    return {"server": servers[server_id]}

@app.get("/servers")
def list_servers(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, description="Maximum number of servers to return")
):
    """Query parameters: comes after ? in the URL."""
    
    # Start with all servers
    filtered_servers = list(servers.values())
    
    # Apply filters
    if environment:
        filtered_servers = [s for s in filtered_servers if s["environment"] == environment]
    
    if status:
        filtered_servers = [s for s in filtered_servers if s["status"] == status]
    
    # Apply limit
    filtered_servers = filtered_servers[:limit]
    
    return {
        "servers": filtered_servers,
        "count": len(filtered_servers),
        "filters": {"environment": environment, "status": status}
    }

@app.get("/servers/{server_id}/logs")
def get_server_logs(
    server_id: str, 
    lines: int = Query(100, description="Number of log lines to return")
):
    """Combine path and query parameters."""
    if server_id not in servers:
        return {"error": "Server not found"}, 404
    
    # Simulate log lines
    fake_logs = [f"Log line {i} from {server_id}" for i in range(1, lines + 1)]
    
    return {
        "server_id": server_id,
        "logs": fake_logs,
        "total_lines": len(fake_logs)
    }
```

### Exercise 5A.5: Parameters
1. Create `parameters.py` and run it
2. Try these URLs:
   - `/servers` (all servers)
   - `/servers?environment=prod` (production servers only)
   - `/servers?status=running&limit=1` (running servers, max 1)
   - `/servers/server_1/logs?lines=5` (5 log lines)
3. Add a new filter for server name (partial match)

## 5A.7 Mini Project: Server Management API

Put it all together:

```python
# server_api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime
import uuid

app = FastAPI(
    title="Server Management API",
    description="A simple API to manage servers",
    version="1.0.0"
)

class ServerCreate(BaseModel):
    name: str
    ip: str
    port: int = 80
    environment: str = "production"
    
    @validator('name')
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError('Name must be at least 3 characters')
        return v
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

class ServerResponse(BaseModel):
    id: str
    name: str
    ip: str
    port: int
    environment: str
    status: str
    created_at: datetime

# In-memory storage
servers: dict = {}

@app.get("/", tags=["General"])
def root():
    """Welcome message."""
    return {
        "message": "Welcome to Server Management API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_servers": len(servers)
    }

@app.post("/servers", response_model=ServerResponse, tags=["Servers"])
def create_server(server: ServerCreate):
    """Create a new server."""
    
    # Check for duplicate names
    for existing in servers.values():
        if existing["name"] == server.name:
            raise HTTPException(
                status_code=400,
                detail=f"Server with name '{server.name}' already exists"
            )
    
    # Create new server
    server_id = str(uuid.uuid4())[:8]  # Short UUID
    server_data = {
        "id": server_id,
        **server.dict(),
        "status": "created",
        "created_at": datetime.now()
    }
    
    servers[server_id] = server_data
    return ServerResponse(**server_data)

@app.get("/servers", response_model=List[ServerResponse], tags=["Servers"])
def list_servers(
    environment: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=100)
):
    """List servers with optional filtering."""
    
    filtered = list(servers.values())
    
    if environment:
        filtered = [s for s in filtered if s["environment"] == environment]
    
    if status:
        filtered = [s for s in filtered if s["status"] == status]
    
    # Apply limit
    filtered = filtered[:limit]
    
    return [ServerResponse(**server) for server in filtered]

@app.get("/servers/{server_id}", response_model=ServerResponse, tags=["Servers"])
def get_server(server_id: str):
    """Get a specific server."""
    if server_id not in servers:
        raise HTTPException(
            status_code=404,
            detail=f"Server with ID '{server_id}' not found"
        )
    
    return ServerResponse(**servers[server_id])

@app.put("/servers/{server_id}/status", tags=["Servers"])
def update_server_status(server_id: str, status: str):
    """Update server status."""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    valid_statuses = ["created", "running", "stopped", "maintenance"]
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {valid_statuses}"
        )
    
    servers[server_id]["status"] = status
    return {"message": f"Server {server_id} status updated to {status}"}

@app.delete("/servers/{server_id}", tags=["Servers"])
def delete_server(server_id: str):
    """Delete a server."""
    if server_id not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    
    deleted_server = servers.pop(server_id)
    return {"message": "Server deleted", "server": deleted_server["name"]}

@app.get("/stats", tags=["Statistics"])
def get_statistics():
    """Get server statistics."""
    if not servers:
        return {"total": 0, "by_status": {}, "by_environment": {}}
    
    # Count by status
    status_counts = {}
    env_counts = {}
    
    for server in servers.values():
        status = server["status"]
        env = server["environment"]
        
        status_counts[status] = status_counts.get(status, 0) + 1
        env_counts[env] = env_counts.get(env, 0) + 1
    
    return {
        "total": len(servers),
        "by_status": status_counts,
        "by_environment": env_counts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Exercise 5A.6: Complete Server API
1. Create `server_api.py` and run it
2. Visit `/docs` and try all the endpoints
3. Create several servers and test filtering
4. Update server statuses and check statistics
5. Try error scenarios (duplicate names, invalid data)

### Challenge Tasks:
1. Add server restart endpoint
2. Add search by server name
3. Add pagination to the server list
4. Store data in a JSON file instead of memory
5. Add timestamps for last_updated

## Key Takeaways

- **Start simple**: Begin with basic endpoints
- **Use Pydantic models**: They provide automatic validation
- **Handle errors properly**: Use HTTPException for API errors
- **Test with /docs**: FastAPI's automatic documentation is your friend
- **Think about data flow**: Request → Validation → Processing → Response

## Next Steps

In Chapter 5B, you'll learn about:
- Database integration
- Authentication and security
- Background tasks
- Deployment strategies