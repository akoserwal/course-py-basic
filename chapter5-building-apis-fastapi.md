# Chapter 5: Building APIs with FastAPI

## Learning Objectives
- Understand REST API design principles for DevOps services
- Build production-ready APIs with FastAPI
- Implement authentication and middleware
- Create health checks and monitoring endpoints
- Design APIs for service discovery and automation
- Deploy and scale API services

## 5.1 FastAPI Fundamentals

### Why FastAPI for DevOps?

FastAPI is ideal for SRE/DevOps applications because it provides:
- **High Performance**: Built on Starlette and Pydantic
- **Automatic Documentation**: OpenAPI (Swagger) integration
- **Type Safety**: Python type hints for better reliability
- **Async Support**: Handle concurrent requests efficiently
- **Easy Testing**: Built-in testing utilities
- **Modern Standards**: OAuth2, JWT, WebSockets support

### Installation and Basic Setup

```bash
# Install FastAPI and dependencies
uv add "fastapi[all]" uvicorn python-multipart
```

### Your First API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn

app = FastAPI(
    title="DevOps API",
    description="API for DevOps automation and monitoring",
    version="1.0.0"
)

# Basic health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DevOps API is running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 5.2 Request and Response Models

### Pydantic Models for Data Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ServerStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class ServerInfo(BaseModel):
    """Model for server information."""
    name: str = Field(..., description="Server name")
    ip_address: str = Field(..., description="Server IP address")
    port: int = Field(default=80, ge=1, le=65535, description="Server port")
    status: ServerStatus = Field(..., description="Server status")
    cpu_usage: float = Field(ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(ge=0, le=100, description="Memory usage percentage")
    last_check: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default=[], description="Server tags")
    
    @validator('ip_address')
    def validate_ip(cls, v):
        import ipaddress
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError('Invalid IP address format')
    
    class Config:
        schema_extra = {
            "example": {
                "name": "web-server-01",
                "ip_address": "192.168.1.10",
                "port": 80,
                "status": "running",
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "tags": ["web", "production"]
            }
        }

class ServerCreate(BaseModel):
    """Model for creating a new server."""
    name: str
    ip_address: str
    port: int = 80
    tags: List[str] = []

class ServerUpdate(BaseModel):
    """Model for updating server information."""
    name: Optional[str] = None
    port: Optional[int] = None
    status: Optional[ServerStatus] = None
    tags: Optional[List[str]] = None

class DeploymentRequest(BaseModel):
    """Model for deployment requests."""
    service_name: str
    version: str
    environment: str = Field(..., regex="^(dev|staging|prod)$")
    config: Dict[str, Any] = Field(default={})
    rollback_enabled: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "service_name": "web-api",
                "version": "v1.2.3",
                "environment": "prod",
                "config": {
                    "replicas": 3,
                    "cpu_limit": "500m",
                    "memory_limit": "1Gi"
                },
                "rollback_enabled": True
            }
        }

class DeploymentStatus(BaseModel):
    """Model for deployment status."""
    deployment_id: str
    service_name: str
    version: str
    environment: str
    status: str
    progress: int = Field(ge=0, le=100)
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
```

## 5.3 Building Server Management API

### In-Memory Data Store (for simplicity)

```python
from typing import Dict, List
import uuid
from datetime import datetime

# In-memory storage (use database in production)
servers_db: Dict[str, ServerInfo] = {}
deployments_db: Dict[str, DeploymentStatus] = {}

def generate_id() -> str:
    """Generate unique ID."""
    return str(uuid.uuid4())
```

### Server Management Endpoints

```python
from fastapi import HTTPException, Query, Path, status

@app.get("/servers", response_model=List[ServerInfo])
async def list_servers(
    status_filter: Optional[ServerStatus] = Query(None, description="Filter by server status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of servers to return")
):
    """List all servers with optional filtering."""
    servers = list(servers_db.values())
    
    # Apply status filter
    if status_filter:
        servers = [s for s in servers if s.status == status_filter]
    
    # Apply tags filter
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        servers = [s for s in servers if any(tag in s.tags for tag in tag_list)]
    
    # Apply limit
    servers = servers[:limit]
    
    return servers

@app.get("/servers/{server_id}", response_model=ServerInfo)
async def get_server(
    server_id: str = Path(..., description="Server ID")
):
    """Get specific server information."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    return servers_db[server_id]

@app.post("/servers", response_model=ServerInfo, status_code=status.HTTP_201_CREATED)
async def create_server(server_data: ServerCreate):
    """Create a new server."""
    # Check if server name already exists
    existing_server = next((s for s in servers_db.values() if s.name == server_data.name), None)
    if existing_server:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Server with name '{server_data.name}' already exists"
        )
    
    # Create new server
    server_id = generate_id()
    server = ServerInfo(
        name=server_data.name,
        ip_address=server_data.ip_address,
        port=server_data.port,
        status=ServerStatus.STOPPED,  # Default status
        cpu_usage=0.0,
        memory_usage=0.0,
        tags=server_data.tags
    )
    
    servers_db[server_id] = server
    return server

@app.put("/servers/{server_id}", response_model=ServerInfo)
async def update_server(
    server_id: str = Path(..., description="Server ID"),
    server_update: ServerUpdate = ...
):
    """Update server information."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    server = servers_db[server_id]
    
    # Update fields if provided
    if server_update.name is not None:
        server.name = server_update.name
    if server_update.port is not None:
        server.port = server_update.port
    if server_update.status is not None:
        server.status = server_update.status
    if server_update.tags is not None:
        server.tags = server_update.tags
    
    server.last_check = datetime.now()
    
    return server

@app.delete("/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_server(
    server_id: str = Path(..., description="Server ID")
):
    """Delete a server."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    del servers_db[server_id]
```

### Server Actions

```python
@app.post("/servers/{server_id}/start")
async def start_server(server_id: str = Path(..., description="Server ID")):
    """Start a server."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    server = servers_db[server_id]
    
    if server.status == ServerStatus.RUNNING:
        return {"message": f"Server {server.name} is already running"}
    
    # Simulate starting server
    server.status = ServerStatus.RUNNING
    server.last_check = datetime.now()
    
    return {"message": f"Server {server.name} started successfully"}

@app.post("/servers/{server_id}/stop")
async def stop_server(server_id: str = Path(..., description="Server ID")):
    """Stop a server."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    server = servers_db[server_id]
    
    if server.status == ServerStatus.STOPPED:
        return {"message": f"Server {server.name} is already stopped"}
    
    # Simulate stopping server
    server.status = ServerStatus.STOPPED
    server.cpu_usage = 0.0
    server.memory_usage = 0.0
    server.last_check = datetime.now()
    
    return {"message": f"Server {server.name} stopped successfully"}

@app.post("/servers/{server_id}/restart")
async def restart_server(server_id: str = Path(..., description="Server ID")):
    """Restart a server."""
    if server_id not in servers_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server {server_id} not found"
        )
    
    server = servers_db[server_id]
    
    # Simulate restart process
    server.status = ServerStatus.MAINTENANCE
    server.last_check = datetime.now()
    
    # In a real implementation, you would:
    # 1. Send stop command
    # 2. Wait for graceful shutdown
    # 3. Send start command
    # 4. Verify server is running
    
    import asyncio
    await asyncio.sleep(1)  # Simulate restart time
    
    server.status = ServerStatus.RUNNING
    server.last_check = datetime.now()
    
    return {"message": f"Server {server.name} restarted successfully"}
```

## 5.4 Deployment Management API

### Deployment Endpoints

```python
import asyncio
from random import uniform

@app.post("/deployments", response_model=DeploymentStatus, status_code=status.HTTP_201_CREATED)
async def create_deployment(deployment_request: DeploymentRequest):
    """Create a new deployment."""
    deployment_id = generate_id()
    
    deployment = DeploymentStatus(
        deployment_id=deployment_id,
        service_name=deployment_request.service_name,
        version=deployment_request.version,
        environment=deployment_request.environment,
        status="pending",
        progress=0,
        started_at=datetime.now()
    )
    
    deployments_db[deployment_id] = deployment
    
    # Start deployment process in background
    asyncio.create_task(simulate_deployment(deployment_id))
    
    return deployment

@app.get("/deployments", response_model=List[DeploymentStatus])
async def list_deployments(
    environment: Optional[str] = Query(None, description="Filter by environment"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of deployments")
):
    """List deployments with optional filtering."""
    deployments = list(deployments_db.values())
    
    # Apply filters
    if environment:
        deployments = [d for d in deployments if d.environment == environment]
    
    if status_filter:
        deployments = [d for d in deployments if d.status == status_filter]
    
    # Sort by started_at (newest first) and apply limit
    deployments.sort(key=lambda x: x.started_at, reverse=True)
    deployments = deployments[:limit]
    
    return deployments

@app.get("/deployments/{deployment_id}", response_model=DeploymentStatus)
async def get_deployment(
    deployment_id: str = Path(..., description="Deployment ID")
):
    """Get deployment status."""
    if deployment_id not in deployments_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )
    
    return deployments_db[deployment_id]

@app.post("/deployments/{deployment_id}/rollback")
async def rollback_deployment(
    deployment_id: str = Path(..., description="Deployment ID")
):
    """Rollback a deployment."""
    if deployment_id not in deployments_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deployment {deployment_id} not found"
        )
    
    deployment = deployments_db[deployment_id]
    
    if deployment.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only rollback completed or failed deployments"
        )
    
    # Create rollback deployment
    rollback_id = generate_id()
    rollback_deployment = DeploymentStatus(
        deployment_id=rollback_id,
        service_name=deployment.service_name,
        version="rollback",
        environment=deployment.environment,
        status="pending",
        progress=0,
        started_at=datetime.now()
    )
    
    deployments_db[rollback_id] = rollback_deployment
    
    # Start rollback process
    asyncio.create_task(simulate_deployment(rollback_id, is_rollback=True))
    
    return {"message": "Rollback initiated", "rollback_id": rollback_id}

async def simulate_deployment(deployment_id: str, is_rollback: bool = False):
    """Simulate deployment process."""
    deployment = deployments_db[deployment_id]
    
    try:
        # Simulate deployment stages
        stages = [
            ("Preparing deployment", 10),
            ("Building application", 30),
            ("Running tests", 50),
            ("Deploying to environment", 80),
            ("Verifying deployment", 100)
        ]
        
        if is_rollback:
            stages = [
                ("Preparing rollback", 20),
                ("Stopping current version", 50),
                ("Starting previous version", 80),
                ("Verifying rollback", 100)
            ]
        
        for stage_name, progress in stages:
            deployment.status = f"running: {stage_name}"
            deployment.progress = progress
            
            # Simulate work
            await asyncio.sleep(uniform(1, 3))
        
        # Complete deployment
        deployment.status = "completed"
        deployment.progress = 100
        deployment.completed_at = datetime.now()
        
    except Exception as e:
        deployment.status = "failed"
        deployment.error_message = str(e)
        deployment.completed_at = datetime.now()
```

## 5.5 Authentication and Security

### API Key Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import hashlib

# Simple API key storage (use database in production)
API_KEYS = {
    "admin": "admin-api-key-12345",
    "deploy": "deploy-api-key-67890",
    "monitor": "monitor-api-key-abcde"
}

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key and return user role."""
    api_key = credentials.credentials
    
    # Find user by API key
    for role, key in API_KEYS.items():
        if secrets.compare_digest(api_key, key):
            return role
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_role(required_role: str):
    """Dependency to require specific role."""
    def role_checker(current_role: str = Depends(verify_api_key)):
        if required_role == "admin" and current_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        elif required_role == "deploy" and current_role not in ["admin", "deploy"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Deploy access required"
            )
        return current_role
    
    return role_checker

# Apply authentication to sensitive endpoints
@app.post("/servers", response_model=ServerInfo, dependencies=[Depends(require_role("admin"))])
async def create_server_protected(server_data: ServerCreate):
    """Create server (admin only)."""
    return await create_server(server_data)

@app.post("/deployments", response_model=DeploymentStatus, dependencies=[Depends(require_role("deploy"))])
async def create_deployment_protected(deployment_request: DeploymentRequest):
    """Create deployment (deploy role required)."""
    return await create_deployment(deployment_request)
```

### JWT Authentication

```python
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

# Configuration
SECRET_KEY = "your-secret-key-here"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User database (use real database in production)
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin"
    },
    "deploy": {
        "username": "deploy",
        "hashed_password": pwd_context.hash("deploy123"),
        "role": "deploy"
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class UserLogin(BaseModel):
    username: str
    password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password."""
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    """Authenticate user credentials."""
    user = users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Authenticate user and return JWT token."""
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
```

## 5.6 Middleware and Monitoring

### Request Logging Middleware

```python
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        # Add response time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

# Add middleware to app
app.add_middleware(RequestLoggingMiddleware)
```

### Metrics Collection

```python
from collections import defaultdict
import time

# Simple metrics storage
metrics = {
    "requests_total": defaultdict(int),
    "request_duration_sum": defaultdict(float),
    "request_duration_count": defaultdict(int),
    "errors_total": defaultdict(int)
}

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get route pattern
        route = request.url.path
        method = request.method
        labels = f"{method} {route}"
        
        try:
            response = await call_next(request)
            
            # Record metrics
            metrics["requests_total"][labels] += 1
            
            duration = time.time() - start_time
            metrics["request_duration_sum"][labels] += duration
            metrics["request_duration_count"][labels] += 1
            
            # Record errors
            if response.status_code >= 400:
                metrics["errors_total"][labels] += 1
            
            return response
        
        except Exception as e:
            # Record error
            metrics["errors_total"][labels] += 1
            duration = time.time() - start_time
            metrics["request_duration_sum"][labels] += duration
            metrics["request_duration_count"][labels] += 1
            
            raise e

app.add_middleware(MetricsMiddleware)

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    lines = []
    
    # Request counts
    for labels, count in metrics["requests_total"].items():
        lines.append(f'http_requests_total{{method_path="{labels}"}} {count}')
    
    # Request duration
    for labels, duration_sum in metrics["request_duration_sum"].items():
        duration_count = metrics["request_duration_count"][labels]
        avg_duration = duration_sum / duration_count if duration_count > 0 else 0
        lines.append(f'http_request_duration_seconds{{method_path="{labels}"}} {avg_duration:.6f}')
    
    # Error counts
    for labels, count in metrics["errors_total"].items():
        lines.append(f'http_errors_total{{method_path="{labels}"}} {count}')
    
    return "\n".join(lines)
```

## Exercise 5: Build a Service Discovery API

### Exercise Overview
Create a comprehensive service discovery API that allows services to register themselves, query for other services, and maintain health status.

### Step 1: Project Setup

```bash
mkdir service-discovery-api
cd service-discovery-api
uv init
uv add "fastapi[all]" uvicorn python-multipart pydantic-settings
mkdir -p {config,tests}
```

### Step 2: Service Models

Create `src/service_discovery/models.py`:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

class ServiceProtocol(str, Enum):
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"

class ServiceEndpoint(BaseModel):
    """Service endpoint information."""
    host: str = Field(..., description="Service host")
    port: int = Field(..., ge=1, le=65535, description="Service port")
    protocol: ServiceProtocol = Field(default=ServiceProtocol.HTTP)
    path: Optional[str] = Field(default="/", description="Health check path")
    
    @property
    def url(self) -> str:
        """Get full URL for the endpoint."""
        if self.protocol in [ServiceProtocol.HTTP, ServiceProtocol.HTTPS]:
            return f"{self.protocol.value}://{self.host}:{self.port}{self.path}"
        return f"{self.host}:{self.port}"

class ServiceRegistration(BaseModel):
    """Service registration request."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment (dev/staging/prod)")
    endpoints: List[ServiceEndpoint] = Field(..., description="Service endpoints")
    tags: List[str] = Field(default=[], description="Service tags")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    health_check_interval: int = Field(default=30, ge=10, le=300, description="Health check interval in seconds")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Service name must be alphanumeric with optional hyphens/underscores')
        return v

class ServiceInfo(ServiceRegistration):
    """Complete service information."""
    service_id: str = Field(..., description="Unique service ID")
    status: ServiceStatus = Field(default=ServiceStatus.HEALTHY)
    registered_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    health_check_url: Optional[str] = None

class ServiceQuery(BaseModel):
    """Service discovery query."""
    name: Optional[str] = None
    environment: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[ServiceStatus] = None
    limit: int = Field(default=100, ge=1, le=1000)

class HealthCheckResult(BaseModel):
    """Health check result."""
    service_id: str
    status: ServiceStatus
    response_time_ms: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
```

### Step 3: Service Discovery API

Create `src/service_discovery/api.py`:

```python
import uuid
import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
import requests

from .models import (
    ServiceRegistration, ServiceInfo, ServiceQuery, 
    HealthCheckResult, ServiceStatus, ServiceEndpoint
)

app = FastAPI(
    title="Service Discovery API",
    description="API for service registration and discovery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use database in production)
services_registry: Dict[str, ServiceInfo] = {}
health_check_results: Dict[str, HealthCheckResult] = {}

def generate_service_id() -> str:
    """Generate unique service ID."""
    return str(uuid.uuid4())

async def perform_health_check(service: ServiceInfo) -> HealthCheckResult:
    """Perform health check on a service."""
    start_time = time.time()
    
    for endpoint in service.endpoints:
        if endpoint.protocol.value in ['http', 'https']:
            try:
                response = requests.get(
                    endpoint.url,
                    timeout=5,
                    headers={'User-Agent': 'ServiceDiscovery/1.0'}
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return HealthCheckResult(
                        service_id=service.service_id,
                        status=ServiceStatus.HEALTHY,
                        response_time_ms=response_time,
                        status_code=response.status_code
                    )
                else:
                    return HealthCheckResult(
                        service_id=service.service_id,
                        status=ServiceStatus.UNHEALTHY,
                        response_time_ms=response_time,
                        status_code=response.status_code,
                        error_message=f"HTTP {response.status_code}"
                    )
            
            except requests.RequestException as e:
                response_time = (time.time() - start_time) * 1000
                return HealthCheckResult(
                    service_id=service.service_id,
                    status=ServiceStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    error_message=str(e)
                )
    
    # If no HTTP endpoints, assume healthy
    return HealthCheckResult(
        service_id=service.service_id,
        status=ServiceStatus.HEALTHY,
        response_time_ms=0
    )

async def health_check_worker():
    """Background worker for health checks."""
    while True:
        current_time = datetime.now()
        
        for service in services_registry.values():
            # Check if health check is due
            if (service.last_health_check is None or 
                current_time - service.last_health_check >= timedelta(seconds=service.health_check_interval)):
                
                # Perform health check
                result = await perform_health_check(service)
                
                # Update service status
                service.status = result.status
                service.last_health_check = current_time
                
                # Store health check result
                health_check_results[service.service_id] = result
        
        # Remove stale services (not seen for 10 minutes)
        stale_cutoff = current_time - timedelta(minutes=10)
        stale_services = [
            service_id for service_id, service in services_registry.items()
            if service.last_seen < stale_cutoff
        ]
        
        for service_id in stale_services:
            del services_registry[service_id]
            if service_id in health_check_results:
                del health_check_results[service_id]
        
        await asyncio.sleep(10)  # Check every 10 seconds

# Start background health check worker
@app.on_event("startup")
async def start_health_checker():
    asyncio.create_task(health_check_worker())

@app.get("/")
async def root():
    """API information."""
    return {
        "service": "Service Discovery API",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /services",
            "discover": "GET /services",
            "health": "GET /services/{service_id}/health"
        }
    }

@app.post("/services", response_model=ServiceInfo, status_code=status.HTTP_201_CREATED)
async def register_service(
    service_registration: ServiceRegistration,
    background_tasks: BackgroundTasks
):
    """Register a new service."""
    # Check if service already exists
    existing_service = None
    for service in services_registry.values():
        if (service.name == service_registration.name and 
            service.environment == service_registration.environment):
            existing_service = service
            break
    
    if existing_service:
        # Update existing service
        existing_service.version = service_registration.version
        existing_service.endpoints = service_registration.endpoints
        existing_service.tags = service_registration.tags
        existing_service.metadata = service_registration.metadata
        existing_service.health_check_interval = service_registration.health_check_interval
        existing_service.last_seen = datetime.now()
        
        return existing_service
    
    # Create new service
    service_id = generate_service_id()
    service_info = ServiceInfo(
        service_id=service_id,
        **service_registration.dict()
    )
    
    # Set health check URL if available
    http_endpoint = next(
        (ep for ep in service_info.endpoints if ep.protocol.value in ['http', 'https']),
        None
    )
    if http_endpoint:
        service_info.health_check_url = http_endpoint.url
    
    services_registry[service_id] = service_info
    
    return service_info

@app.get("/services", response_model=List[ServiceInfo])
async def discover_services(
    name: Optional[str] = None,
    environment: Optional[str] = None,
    tags: Optional[str] = None,
    status_filter: Optional[ServiceStatus] = None,
    limit: int = 100
):
    """Discover services based on criteria."""
    services = list(services_registry.values())
    
    # Apply filters
    if name:
        services = [s for s in services if name.lower() in s.name.lower()]
    
    if environment:
        services = [s for s in services if s.environment == environment]
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        services = [s for s in services if any(tag in s.tags for tag in tag_list)]
    
    if status_filter:
        services = [s for s in services if s.status == status_filter]
    
    # Sort by registration time (newest first) and apply limit
    services.sort(key=lambda x: x.registered_at, reverse=True)
    services = services[:limit]
    
    return services

@app.get("/services/{service_id}", response_model=ServiceInfo)
async def get_service(service_id: str):
    """Get specific service information."""
    if service_id not in services_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_id} not found"
        )
    
    return services_registry[service_id]

@app.delete("/services/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_service(service_id: str):
    """Unregister a service."""
    if service_id not in services_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_id} not found"
        )
    
    del services_registry[service_id]
    if service_id in health_check_results:
        del health_check_results[service_id]

@app.post("/services/{service_id}/heartbeat")
async def service_heartbeat(service_id: str):
    """Update service last seen timestamp."""
    if service_id not in services_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_id} not found"
        )
    
    services_registry[service_id].last_seen = datetime.now()
    return {"message": "Heartbeat received"}

@app.get("/services/{service_id}/health", response_model=HealthCheckResult)
async def get_service_health(service_id: str):
    """Get service health status."""
    if service_id not in services_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service {service_id} not found"
        )
    
    if service_id not in health_check_results:
        # Perform immediate health check
        service = services_registry[service_id]
        result = await perform_health_check(service)
        health_check_results[service_id] = result
    
    return health_check_results[service_id]

@app.get("/health")
async def api_health():
    """Service discovery API health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services_count": len(services_registry),
        "healthy_services": len([s for s in services_registry.values() if s.status == ServiceStatus.HEALTHY]),
        "unhealthy_services": len([s for s in services_registry.values() if s.status == ServiceStatus.UNHEALTHY])
    }

@app.get("/stats")
async def get_stats():
    """Get service discovery statistics."""
    services = list(services_registry.values())
    
    # Group by environment
    env_counts = {}
    for service in services:
        env_counts[service.environment] = env_counts.get(service.environment, 0) + 1
    
    # Group by status
    status_counts = {}
    for service in services:
        status_counts[service.status.value] = status_counts.get(service.status.value, 0) + 1
    
    return {
        "total_services": len(services),
        "by_environment": env_counts,
        "by_status": status_counts,
        "latest_registrations": [
            {
                "name": s.name,
                "environment": s.environment,
                "registered_at": s.registered_at.isoformat()
            }
            for s in sorted(services, key=lambda x: x.registered_at, reverse=True)[:5]
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4: Client SDK

Create `src/service_discovery/client.py`:

```python
import requests
from typing import List, Optional
from .models import ServiceRegistration, ServiceInfo, ServiceStatus

class ServiceDiscoveryClient:
    """Client SDK for service discovery API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def register(self, service: ServiceRegistration) -> ServiceInfo:
        """Register a service."""
        response = self.session.post(
            f"{self.base_url}/services",
            json=service.dict()
        )
        response.raise_for_status()
        return ServiceInfo(**response.json())
    
    def discover(self, 
                name: Optional[str] = None,
                environment: Optional[str] = None,
                tags: Optional[str] = None,
                status: Optional[ServiceStatus] = None) -> List[ServiceInfo]:
        """Discover services."""
        params = {}
        if name:
            params['name'] = name
        if environment:
            params['environment'] = environment
        if tags:
            params['tags'] = tags
        if status:
            params['status_filter'] = status.value
        
        response = self.session.get(
            f"{self.base_url}/services",
            params=params
        )
        response.raise_for_status()
        return [ServiceInfo(**service) for service in response.json()]
    
    def get_service(self, service_id: str) -> ServiceInfo:
        """Get specific service."""
        response = self.session.get(f"{self.base_url}/services/{service_id}")
        response.raise_for_status()
        return ServiceInfo(**response.json())
    
    def unregister(self, service_id: str):
        """Unregister a service."""
        response = self.session.delete(f"{self.base_url}/services/{service_id}")
        response.raise_for_status()
    
    def heartbeat(self, service_id: str):
        """Send heartbeat for a service."""
        response = self.session.post(f"{self.base_url}/services/{service_id}/heartbeat")
        response.raise_for_status()
    
    def health_check(self, service_id: str):
        """Get service health status."""
        response = self.session.get(f"{self.base_url}/services/{service_id}/health")
        response.raise_for_status()
        return response.json()
```

### Step 5: Exercise Tasks

1. **Start the service discovery API:**
   ```bash
   uv run python -m service_discovery.api
   ```

2. **Test the API with curl:**
   ```bash
   # Register a service
   curl -X POST "http://localhost:8000/services" \
        -H "Content-Type: application/json" \
        -d '{
          "name": "web-api",
          "version": "1.0.0",
          "environment": "prod",
          "endpoints": [
            {
              "host": "api.example.com",
              "port": 80,
              "protocol": "http",
              "path": "/health"
            }
          ],
          "tags": ["web", "api"]
        }'
   
   # Discover services
   curl "http://localhost:8000/services?environment=prod"
   ```

3. **Create a demo service that registers itself:**
   ```python
   # demo_service.py
   import time
   import asyncio
   from service_discovery.client import ServiceDiscoveryClient
   from service_discovery.models import ServiceRegistration, ServiceEndpoint, ServiceProtocol
   
   async def main():
       client = ServiceDiscoveryClient()
       
       # Register this service
       registration = ServiceRegistration(
           name="demo-service",
           version="1.0.0", 
           environment="dev",
           endpoints=[
               ServiceEndpoint(
                   host="localhost",
                   port=8001,
                   protocol=ServiceProtocol.HTTP,
                   path="/health"
               )
           ],
           tags=["demo", "test"]
       )
       
       service_info = client.register(registration)
       print(f"Registered service: {service_info.service_id}")
       
       # Send periodic heartbeats
       while True:
           await asyncio.sleep(30)
           client.heartbeat(service_info.service_id)
           print("Heartbeat sent")
   
   if __name__ == "__main__":
       asyncio.run(main())
   ```

## Key Takeaways

- FastAPI provides excellent performance and developer experience for API development
- Pydantic models ensure data validation and automatic documentation
- Authentication and authorization are critical for production APIs
- Middleware enables cross-cutting concerns like logging and metrics
- Background tasks handle asynchronous operations
- Service discovery is essential for microservice architectures
- Health checks and monitoring ensure API reliability

This foundation in API development with FastAPI prepares you to build robust, scalable services for DevOps automation and infrastructure management.