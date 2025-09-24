#!/usr/bin/env python3
"""
Service Mesh API - Advanced FastAPI implementation
Demonstrates service discovery, load balancing, and circuit breaker patterns.
"""

import asyncio
import time
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import httpx
import uvicorn

# Models
class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

class ServiceInstance(BaseModel):
    id: str
    name: str
    host: str
    port: int
    weight: int = Field(default=100, ge=1, le=1000)
    status: ServiceStatus = ServiceStatus.HEALTHY
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    failure_count: int = 0
    
    @validator('health_check_url', pre=True, always=True)
    def set_health_check_url(cls, v, values):
        if v is None and 'host' in values and 'port' in values:
            return f"http://{values['host']}:{values['port']}/health"
        return v
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

class ServiceRegistration(BaseModel):
    name: str
    host: str
    port: int
    weight: int = 100
    metadata: Dict[str, Any] = Field(default_factory=dict)
    health_check_url: Optional[str] = None

class HealthCheckResult(BaseModel):
    service_id: str
    status: ServiceStatus
    response_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None

class ProxyRequest(BaseModel):
    method: str = "GET"
    path: str = "/"
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None
    timeout: int = 30

class LoadBalancerStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    uptime_seconds: float

# Circuit Breaker Implementation
@dataclass
class CircuitBreakerState:
    failure_threshold: int = 5
    recovery_timeout: int = 30
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        return self.state == "OPEN"
    
    def is_half_open(self) -> bool:
        return self.state == "HALF_OPEN"
    
    def should_attempt_reset(self) -> bool:
        if self.state == "OPEN" and self.last_failure_time:
            return datetime.utcnow() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Service Registry
class ServiceRegistry:
    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.request_stats: Dict[str, Dict[str, Any]] = {}
        
    def register_service(self, registration: ServiceRegistration) -> ServiceInstance:
        """Register a new service instance."""
        service_id = self._generate_service_id(registration.name, registration.host, registration.port)
        
        instance = ServiceInstance(
            id=service_id,
            name=registration.name,
            host=registration.host,
            port=registration.port,
            weight=registration.weight,
            metadata=registration.metadata,
            health_check_url=registration.health_check_url
        )
        
        self.services[service_id] = instance
        self.circuit_breakers[service_id] = CircuitBreakerState()
        self.request_stats[service_id] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0
        }
        
        return instance
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance."""
        if service_id in self.services:
            del self.services[service_id]
            self.circuit_breakers.pop(service_id, None)
            self.request_stats.pop(service_id, None)
            return True
        return False
    
    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """Get a specific service instance."""
        return self.services.get(service_id)
    
    def get_services_by_name(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances of a service by name."""
        return [svc for svc in self.services.values() if svc.name == service_name]
    
    def get_healthy_services(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances of a service."""
        services = self.get_services_by_name(service_name)
        healthy_services = []
        
        for service in services:
            circuit_breaker = self.circuit_breakers.get(service.id)
            if circuit_breaker and circuit_breaker.is_open():
                if circuit_breaker.should_attempt_reset():
                    circuit_breaker.state = "HALF_OPEN"
                else:
                    continue
            
            if service.is_healthy:
                healthy_services.append(service)
        
        return healthy_services
    
    def update_service_health(self, service_id: str, status: ServiceStatus):
        """Update service health status."""
        if service_id in self.services:
            self.services[service_id].status = status
            self.services[service_id].last_seen = datetime.utcnow()
    
    def heartbeat(self, service_id: str):
        """Update service last seen timestamp."""
        if service_id in self.services:
            self.services[service_id].last_seen = datetime.utcnow()
    
    def record_request_result(self, service_id: str, success: bool, response_time: float):
        """Record request statistics and update circuit breaker."""
        if service_id not in self.request_stats:
            return
        
        stats = self.request_stats[service_id]
        stats["total_requests"] += 1
        stats["total_response_time"] += response_time
        
        circuit_breaker = self.circuit_breakers.get(service_id)
        
        if success:
            stats["successful_requests"] += 1
            if circuit_breaker:
                circuit_breaker.record_success()
        else:
            stats["failed_requests"] += 1
            if circuit_breaker:
                circuit_breaker.record_failure()
    
    def get_service_stats(self, service_id: str) -> Dict[str, Any]:
        """Get service statistics."""
        stats = self.request_stats.get(service_id, {})
        circuit_breaker = self.circuit_breakers.get(service_id)
        
        total_requests = stats.get("total_requests", 0)
        avg_response_time = (stats.get("total_response_time", 0) / total_requests) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": stats.get("successful_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "avg_response_time_ms": avg_response_time,
            "circuit_breaker_state": circuit_breaker.state if circuit_breaker else "UNKNOWN",
            "failure_count": circuit_breaker.failure_count if circuit_breaker else 0
        }
    
    @staticmethod
    def _generate_service_id(name: str, host: str, port: int) -> str:
        """Generate unique service ID."""
        unique_string = f"{name}:{host}:{port}:{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

# Load Balancer
class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = {}
        self.consistent_hash_ring: Dict[str, List[str]] = {}
    
    def select_service(self, services: List[ServiceInstance], request_id: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select a service instance based on load balancing strategy."""
        if not services:
            return None
        
        if len(services) == 1:
            return services[0]
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(services)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(services)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(services)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(services)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(services, request_id or "")
        else:
            return services[0]
    
    def _round_robin_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection."""
        service_group = services[0].name
        
        if service_group not in self.round_robin_counters:
            self.round_robin_counters[service_group] = 0
        
        selected_index = self.round_robin_counters[service_group] % len(services)
        self.round_robin_counters[service_group] += 1
        
        return services[selected_index]
    
    def _weighted_round_robin_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin selection."""
        total_weight = sum(service.weight for service in services)
        
        if total_weight == 0:
            return random.choice(services)
        
        random_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for service in services:
            current_weight += service.weight
            if random_weight <= current_weight:
                return service
        
        return services[-1]
    
    def _least_connections_select(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection (simplified)."""
        # In a real implementation, you'd track active connections
        # Here we'll use failure count as a proxy
        return min(services, key=lambda s: s.failure_count)
    
    def _consistent_hash_select(self, services: List[ServiceInstance], key: str) -> ServiceInstance:
        """Consistent hash selection."""
        service_group = services[0].name
        
        # Build hash ring if not exists
        if service_group not in self.consistent_hash_ring:
            ring = []
            for service in services:
                # Add multiple virtual nodes for each service
                for i in range(3):  # 3 virtual nodes per service
                    virtual_key = f"{service.id}:{i}"
                    ring.append(virtual_key)
            
            ring.sort()
            self.consistent_hash_ring[service_group] = ring
        
        # Hash the key and find the appropriate service
        key_hash = hashlib.md5(key.encode()).hexdigest()
        ring = self.consistent_hash_ring[service_group]
        
        for virtual_key in ring:
            if virtual_key >= key_hash:
                service_id = virtual_key.split(':')[0]
                return next(s for s in services if s.id == service_id)
        
        # Wrap around to first service
        service_id = ring[0].split(':')[0]
        return next(s for s in services if s.id == service_id)

# FastAPI Application
app = FastAPI(
    title="Service Mesh API",
    description="Advanced service discovery and load balancing API",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Global instances
registry = ServiceRegistry()
load_balancer = LoadBalancer()
security = HTTPBearer(auto_error=False)

# Health Check Worker
class HealthChecker:
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.client = httpx.AsyncClient(timeout=30.0)
        self.running = False
    
    async def start(self):
        """Start health check worker."""
        self.running = True
        while self.running:
            await self._check_all_services()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def stop(self):
        """Stop health check worker."""
        self.running = False
        await self.client.aclose()
    
    async def _check_all_services(self):
        """Check health of all registered services."""
        services = list(self.registry.services.values())
        
        if not services:
            return
        
        # Check services concurrently
        tasks = [self._check_service_health(service) for service in services]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: ServiceInstance):
        """Check health of a single service."""
        start_time = time.time()
        
        try:
            response = await self.client.get(service.health_check_url, timeout=10.0)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.registry.update_service_health(service.id, ServiceStatus.HEALTHY)
                self.registry.record_request_result(service.id, True, response_time)
            else:
                self.registry.update_service_health(service.id, ServiceStatus.UNHEALTHY)
                self.registry.record_request_result(service.id, False, response_time)
        
        except Exception:
            response_time = (time.time() - start_time) * 1000
            self.registry.update_service_health(service.id, ServiceStatus.UNHEALTHY)
            self.registry.record_request_result(service.id, False, response_time)

# Initialize health checker
health_checker = HealthChecker(registry)

@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    asyncio.create_task(health_checker.start())

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up background tasks."""
    await health_checker.stop()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token authentication."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # In production, validate the token properly
    if credentials.credentials != "demo-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return {"user": "demo"}

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Service Mesh API",
        "version": "1.0.0",
        "description": "Advanced service discovery and load balancing",
        "total_services": len(registry.services),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy_services = [s for s in registry.services.values() if s.is_healthy]
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "total": len(registry.services),
            "healthy": len(healthy_services),
            "unhealthy": len(registry.services) - len(healthy_services)
        }
    }

@app.post("/services/register", response_model=ServiceInstance)
async def register_service(
    registration: ServiceRegistration,
    user = Depends(get_current_user)
):
    """Register a new service instance."""
    instance = registry.register_service(registration)
    return instance

@app.delete("/services/{service_id}")
async def deregister_service(
    service_id: str,
    user = Depends(get_current_user)
):
    """Deregister a service instance."""
    if registry.deregister_service(service_id):
        return {"message": f"Service {service_id} deregistered successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )

@app.get("/services", response_model=List[ServiceInstance])
async def list_services(
    service_name: Optional[str] = None,
    status_filter: Optional[ServiceStatus] = None
):
    """List all registered services."""
    services = list(registry.services.values())
    
    if service_name:
        services = [s for s in services if s.name == service_name]
    
    if status_filter:
        services = [s for s in services if s.status == status_filter]
    
    return services

@app.get("/services/{service_id}", response_model=ServiceInstance)
async def get_service(service_id: str):
    """Get a specific service instance."""
    service = registry.get_service(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    return service

@app.post("/services/{service_id}/heartbeat")
async def service_heartbeat(service_id: str):
    """Update service heartbeat."""
    service = registry.get_service(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    registry.heartbeat(service_id)
    return {"message": "Heartbeat received", "timestamp": datetime.utcnow().isoformat()}

@app.get("/services/{service_name}/discover")
async def discover_service(
    service_name: str,
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    request_id: Optional[str] = None
):
    """Discover and select a service instance."""
    healthy_services = registry.get_healthy_services(service_name)
    
    if not healthy_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No healthy instances of service '{service_name}' available"
        )
    
    # Update load balancer strategy if needed
    load_balancer.strategy = strategy
    
    selected_service = load_balancer.select_service(healthy_services, request_id)
    
    if not selected_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to select service instance"
        )
    
    return {
        "service": selected_service,
        "selection_strategy": strategy,
        "total_healthy_instances": len(healthy_services),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/proxy/{service_name}")
async def proxy_request(
    service_name: str,
    proxy_request: ProxyRequest,
    request: Request,
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
):
    """Proxy request to a service instance."""
    # Generate request ID for consistent hashing
    request_id = request.headers.get("X-Request-ID", str(hash(request.client.host + str(time.time()))))
    
    healthy_services = registry.get_healthy_services(service_name)
    
    if not healthy_services:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No healthy instances of service '{service_name}' available"
        )
    
    # Select service instance
    load_balancer.strategy = strategy
    selected_service = load_balancer.select_service(healthy_services, request_id)
    
    if not selected_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to select service instance"
        )
    
    # Check circuit breaker
    circuit_breaker = registry.circuit_breakers.get(selected_service.id)
    if circuit_breaker and circuit_breaker.is_open() and not circuit_breaker.should_attempt_reset():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service circuit breaker is open"
        )
    
    # Make proxied request
    start_time = time.time()
    target_url = f"{selected_service.url}{proxy_request.path}"
    
    try:
        async with httpx.AsyncClient(timeout=proxy_request.timeout) as client:
            response = await client.request(
                method=proxy_request.method,
                url=target_url,
                headers=proxy_request.headers,
                content=proxy_request.body,
            )
            
            response_time = (time.time() - start_time) * 1000
            registry.record_request_result(selected_service.id, True, response_time)
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "response_time_ms": response_time,
                "selected_service": selected_service.id,
                "request_id": request_id
            }
    
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        registry.record_request_result(selected_service.id, False, response_time)
        
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Proxy request failed: {str(e)}"
        )

@app.get("/services/{service_id}/stats")
async def get_service_stats(service_id: str):
    """Get service statistics."""
    service = registry.get_service(service_id)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    stats = registry.get_service_stats(service_id)
    return {
        "service_id": service_id,
        "service_name": service.name,
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/stats/overview")
async def get_overview_stats():
    """Get overall system statistics."""
    total_services = len(registry.services)
    healthy_services = len([s for s in registry.services.values() if s.is_healthy])
    
    total_requests = sum(
        stats.get("total_requests", 0) 
        for stats in registry.request_stats.values()
    )
    
    successful_requests = sum(
        stats.get("successful_requests", 0) 
        for stats in registry.request_stats.values()
    )
    
    return {
        "services": {
            "total": total_services,
            "healthy": healthy_services,
            "unhealthy": total_services - healthy_services
        },
        "requests": {
            "total": total_requests,
            "successful": successful_requests,
            "failed": total_requests - successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0
        },
        "load_balancer": {
            "strategy": load_balancer.strategy.value
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "service_mesh_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )