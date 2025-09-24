# Final Project: Complete DevOps Automation Platform

## Project Overview

Build a comprehensive DevOps automation platform that demonstrates all the skills learned throughout the course. This platform will include infrastructure monitoring, service discovery, automated deployments, alerting, and a web dashboard for managing everything.

## Learning Objectives

- Integrate all course concepts into a single, cohesive project
- Demonstrate real-world DevOps automation scenarios
- Build a production-ready system with proper testing and monitoring
- Practice project architecture and system design
- Create documentation and deployment guides

## Project Architecture

```
DevOps Automation Platform
├── Core API (FastAPI)
│   ├── Service Discovery
│   ├── Infrastructure Monitoring
│   ├── Deployment Management
│   └── Alert Management
├── Worker Services
│   ├── Health Check Workers
│   ├── Deployment Workers
│   └── Metrics Collectors
├── Web Dashboard (Optional)
├── Database (SQLite/PostgreSQL)
├── Message Queue (Redis)
└── External Integrations
    ├── Slack/Discord
    ├── Docker Registry
    └── Kubernetes/Docker
```

## Phase 1: Foundation Setup

### Step 1: Project Structure

```bash
mkdir devops-automation-platform
cd devops-automation-platform
uv init
mkdir -p {src/platform/{api,workers,models,utils},web,config,scripts,tests/{unit,integration},docs,k8s}
```

### Step 2: Core Dependencies

```bash
uv add "fastapi[all]" uvicorn sqlalchemy alembic redis celery pydantic-settings
uv add requests pyyaml click psutil docker kubernetes
uv add --dev pytest pytest-cov pytest-asyncio httpx black flake8 mypy
```

### Step 3: Configuration Management

Create `src/platform/config.py`:

```python
from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "DevOps Automation Platform"
    api_version: str = "1.0.0"
    
    # Database
    database_url: str = "sqlite:///./devops_platform.db"
    database_echo: bool = False
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Authentication
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Monitoring
    health_check_interval: int = 30
    metrics_retention_days: int = 7
    
    # Alerting
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    alert_recipients: List[str] = []
    
    # External Services
    docker_registry_url: Optional[str] = None
    docker_registry_username: Optional[str] = None
    docker_registry_password: Optional[str] = None
    
    # Kubernetes
    kubeconfig_path: Optional[str] = None
    kubernetes_namespace: str = "default"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

## Phase 2: Database Models and Core API

### Database Models

Create `src/platform/models/database.py`:

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()

class Service(Base):
    """Service registry model."""
    __tablename__ = "services"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    url = Column(String(512), nullable=False)
    environment = Column(String(50), nullable=False, index=True)
    version = Column(String(50), nullable=True)
    status = Column(String(20), default="unknown", index=True)
    tags = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_seen = Column(DateTime, default=func.now())
    
    # Relationships
    health_checks = relationship("HealthCheck", back_populates="service")
    deployments = relationship("Deployment", back_populates="service")

class HealthCheck(Base):
    """Health check results."""
    __tablename__ = "health_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    response_time_ms = Column(Float, nullable=True)
    status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    checked_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    service = relationship("Service", back_populates="health_checks")

class Deployment(Base):
    """Deployment tracking."""
    __tablename__ = "deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    service_id = Column(Integer, ForeignKey("services.id"), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    environment = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)
    config = Column(JSON, default=dict)
    started_at = Column(DateTime, default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    service = relationship("Service", back_populates="deployments")

class Alert(Base):
    """Alert tracking."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    service_name = Column(String(255), nullable=True, index=True)
    status = Column(String(20), default="open", index=True)
    created_at = Column(DateTime, default=func.now(), index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, default=dict)

class Metric(Base):
    """System metrics."""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    value = Column(Float, nullable=False)
    labels = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=func.now(), index=True)

class InfrastructureNode(Base):
    """Infrastructure node tracking."""
    __tablename__ = "infrastructure_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    type = Column(String(50), nullable=False)  # server, container, vm
    ip_address = Column(String(45), nullable=True)
    hostname = Column(String(255), nullable=True)
    status = Column(String(20), default="unknown", index=True)
    cpu_cores = Column(Integer, nullable=True)
    memory_gb = Column(Float, nullable=True)
    disk_gb = Column(Float, nullable=True)
    os_info = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())
```

### Database Setup

Create `src/platform/database.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models.database import Base
from .config import settings
from typing import Generator

# Create engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database."""
    create_tables()
```

## Phase 3: Core API Implementation

### Service Management API

Create `src/platform/api/services.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..models.database import Service, HealthCheck
from ..models.schemas import ServiceCreate, ServiceResponse, ServiceUpdate, HealthCheckResponse

router = APIRouter(prefix="/services", tags=["services"])

@router.post("/", response_model=ServiceResponse, status_code=status.HTTP_201_CREATED)
async def create_service(
    service_data: ServiceCreate,
    db: Session = Depends(get_db)
):
    """Register a new service."""
    # Check if service already exists
    existing = db.query(Service).filter(Service.name == service_data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Service '{service_data.name}' already exists"
        )
    
    # Create new service
    service = Service(
        name=service_data.name,
        url=service_data.url,
        environment=service_data.environment,
        version=service_data.version,
        tags=service_data.tags,
        metadata=service_data.metadata
    )
    
    db.add(service)
    db.commit()
    db.refresh(service)
    
    return ServiceResponse.from_orm(service)

@router.get("/", response_model=List[ServiceResponse])
async def list_services(
    environment: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    skip: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List services with filtering."""
    query = db.query(Service)
    
    # Apply filters
    if environment:
        query = query.filter(Service.environment == environment)
    
    if status_filter:
        query = query.filter(Service.status == status_filter)
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        for tag in tag_list:
            query = query.filter(Service.tags.contains([tag]))
    
    # Apply pagination
    services = query.offset(skip).limit(limit).all()
    
    return [ServiceResponse.from_orm(service) for service in services]

@router.get("/{service_id}", response_model=ServiceResponse)
async def get_service(
    service_id: int,
    db: Session = Depends(get_db)
):
    """Get service by ID."""
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    return ServiceResponse.from_orm(service)

@router.put("/{service_id}", response_model=ServiceResponse)
async def update_service(
    service_id: int,
    service_data: ServiceUpdate,
    db: Session = Depends(get_db)
):
    """Update service."""
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    # Update fields
    for field, value in service_data.dict(exclude_unset=True).items():
        setattr(service, field, value)
    
    service.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(service)
    
    return ServiceResponse.from_orm(service)

@router.delete("/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_service(
    service_id: int,
    db: Session = Depends(get_db)
):
    """Delete service."""
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    db.delete(service)
    db.commit()

@router.post("/{service_id}/heartbeat")
async def service_heartbeat(
    service_id: int,
    db: Session = Depends(get_db)
):
    """Update service last seen timestamp."""
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    service.last_seen = datetime.utcnow()
    db.commit()
    
    return {"message": "Heartbeat received"}

@router.get("/{service_id}/health", response_model=List[HealthCheckResponse])
async def get_service_health_history(
    service_id: int,
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db)
):
    """Get service health check history."""
    service = db.query(Service).filter(Service.id == service_id).first()
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    health_checks = (
        db.query(HealthCheck)
        .filter(HealthCheck.service_id == service_id)
        .order_by(HealthCheck.checked_at.desc())
        .limit(limit)
        .all()
    )
    
    return [HealthCheckResponse.from_orm(hc) for hc in health_checks]
```

### Monitoring API

Create `src/platform/api/monitoring.py`:

```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, func
from sqlalchemy import and_, desc
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models.database import Service, HealthCheck, Metric, InfrastructureNode
from ..models.schemas import MonitoringDashboard, HealthSummary, MetricResponse

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/dashboard", response_model=MonitoringDashboard)
async def get_monitoring_dashboard(
    db: Session = Depends(get_db)
):
    """Get monitoring dashboard data."""
    # Service statistics
    total_services = db.query(Service).count()
    healthy_services = db.query(Service).filter(Service.status == "healthy").count()
    unhealthy_services = db.query(Service).filter(Service.status == "unhealthy").count()
    
    # Recent health checks
    recent_health_checks = (
        db.query(HealthCheck)
        .join(Service)
        .order_by(HealthCheck.checked_at.desc())
        .limit(10)
        .all()
    )
    
    # Infrastructure nodes
    total_nodes = db.query(InfrastructureNode).count()
    active_nodes = db.query(InfrastructureNode).filter(
        InfrastructureNode.status == "active"
    ).count()
    
    # Response time trends (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(hours=24)
    avg_response_times = (
        db.query(
            func.date_trunc('hour', HealthCheck.checked_at).label('hour'),
            func.avg(HealthCheck.response_time_ms).label('avg_response_time')
        )
        .filter(HealthCheck.checked_at >= yesterday)
        .group_by(func.date_trunc('hour', HealthCheck.checked_at))
        .order_by(func.date_trunc('hour', HealthCheck.checked_at))
        .all()
    )
    
    return MonitoringDashboard(
        service_stats={
            "total": total_services,
            "healthy": healthy_services,
            "unhealthy": unhealthy_services,
            "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0
        },
        infrastructure_stats={
            "total_nodes": total_nodes,
            "active_nodes": active_nodes
        },
        recent_health_checks=[
            {
                "service_name": hc.service.name,
                "status": hc.status,
                "response_time_ms": hc.response_time_ms,
                "checked_at": hc.checked_at
            }
            for hc in recent_health_checks
        ],
        response_time_trends=[
            {
                "timestamp": result.hour,
                "avg_response_time_ms": float(result.avg_response_time) if result.avg_response_time else 0
            }
            for result in avg_response_times
        ]
    )

@router.get("/health-summary", response_model=HealthSummary)
async def get_health_summary(
    environment: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get overall health summary."""
    query = db.query(Service)
    if environment:
        query = query.filter(Service.environment == environment)
    
    services = query.all()
    
    # Calculate health statistics
    total = len(services)
    healthy = sum(1 for s in services if s.status == "healthy")
    unhealthy = sum(1 for s in services if s.status == "unhealthy")
    unknown = sum(1 for s in services if s.status == "unknown")
    
    # Get recent incidents (health failures in last hour)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_incidents = (
        db.query(HealthCheck)
        .join(Service)
        .filter(
            and_(
                HealthCheck.status == "unhealthy",
                HealthCheck.checked_at >= one_hour_ago
            )
        )
        .count()
    )
    
    return HealthSummary(
        total_services=total,
        healthy_services=healthy,
        unhealthy_services=unhealthy,
        unknown_services=unknown,
        health_percentage=(healthy / total * 100) if total > 0 else 0,
        recent_incidents=recent_incidents,
        last_updated=datetime.utcnow()
    )

@router.get("/metrics", response_model=List[MetricResponse])
async def get_metrics(
    name: Optional[str] = Query(None),
    hours: int = Query(24, le=168),  # Max 7 days
    limit: int = Query(1000, le=10000),
    db: Session = Depends(get_db)
):
    """Get system metrics."""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    query = db.query(Metric).filter(Metric.timestamp >= cutoff_time)
    
    if name:
        query = query.filter(Metric.name == name)
    
    metrics = query.order_by(desc(Metric.timestamp)).limit(limit).all()
    
    return [MetricResponse.from_orm(metric) for metric in metrics]

@router.get("/infrastructure")
async def get_infrastructure_status(
    db: Session = Depends(get_db)
):
    """Get infrastructure node status."""
    nodes = db.query(InfrastructureNode).all()
    
    return {
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "status": node.status,
                "ip_address": node.ip_address,
                "cpu_cores": node.cpu_cores,
                "memory_gb": node.memory_gb,
                "last_seen": node.last_seen
            }
            for node in nodes
        ],
        "summary": {
            "total": len(nodes),
            "active": sum(1 for n in nodes if n.status == "active"),
            "inactive": sum(1 for n in nodes if n.status == "inactive"),
            "unknown": sum(1 for n in nodes if n.status == "unknown")
        }
    }
```

## Phase 4: Worker Services

### Health Check Worker

Create `src/platform/workers/health_checker.py`:

```python
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Dict, Any

from ..models.database import Service, HealthCheck
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class HealthCheckWorker:
    """Background worker for health checks."""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def check_service_health(self, service: Service) -> HealthCheck:
        """Perform health check on a single service."""
        start_time = time.time()
        
        try:
            health_url = f"{service.url.rstrip('/')}/health"
            
            async with self.session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000
                
                # Determine health status
                if response.status == 200:
                    status = "healthy"
                elif 200 <= response.status < 400:
                    status = "warning"
                else:
                    status = "unhealthy"
                
                return HealthCheck(
                    service_id=service.id,
                    status=status,
                    response_time_ms=response_time,
                    status_code=response.status,
                    checked_at=datetime.utcnow()
                )
        
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                service_id=service.id,
                status="unhealthy",
                response_time_ms=response_time,
                error_message="Request timeout",
                checked_at=datetime.utcnow()
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                service_id=service.id,
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                checked_at=datetime.utcnow()
            )
    
    async def check_all_services(self):
        """Check health of all registered services."""
        db = self.SessionLocal()
        try:
            # Get services that need health checks
            cutoff_time = datetime.utcnow() - timedelta(seconds=settings.health_check_interval)
            
            services = db.query(Service).filter(
                Service.last_seen >= cutoff_time - timedelta(minutes=10)  # Only check recently active services
            ).all()
            
            if not services:
                logger.info("No services to check")
                return
            
            logger.info(f"Checking health of {len(services)} services")
            
            # Perform health checks concurrently
            tasks = [self.check_service_health(service) for service in services]
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Save results and update service status
            for i, result in enumerate(health_results):
                if isinstance(result, Exception):
                    logger.error(f"Health check failed for {services[i].name}: {result}")
                    continue
                
                # Save health check result
                db.add(result)
                
                # Update service status
                service = services[i]
                old_status = service.status
                service.status = result.status
                service.updated_at = datetime.utcnow()
                
                # Log status changes
                if old_status != service.status:
                    logger.info(f"Service {service.name} status changed: {old_status} -> {service.status}")
            
            db.commit()
            logger.info(f"Completed health checks for {len(services)} services")
        
        except Exception as e:
            logger.error(f"Error during health check cycle: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def cleanup_old_health_checks(self):
        """Remove old health check records."""
        db = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=settings.metrics_retention_days)
            
            deleted_count = db.query(HealthCheck).filter(
                HealthCheck.checked_at < cutoff_date
            ).delete()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old health check records")
                db.commit()
        
        except Exception as e:
            logger.error(f"Error during health check cleanup: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def run(self):
        """Main worker loop."""
        logger.info("Starting health check worker")
        
        try:
            while True:
                # Perform health checks
                await self.check_all_services()
                
                # Cleanup old records every hour
                if datetime.utcnow().minute == 0:
                    await self.cleanup_old_health_checks()
                
                # Wait before next cycle
                await asyncio.sleep(settings.health_check_interval)
        
        except KeyboardInterrupt:
            logger.info("Health check worker stopped")
        finally:
            await self.session.close()

async def main():
    """Run health check worker."""
    worker = HealthCheckWorker()
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Metrics Collector Worker

Create `src/platform/workers/metrics_collector.py`:

```python
import asyncio
import psutil
import docker
import subprocess
from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import Dict, Any, List

from ..models.database import Metric, InfrastructureNode
from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """Collect system and application metrics."""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Could not connect to Docker: {e}")
            self.docker_client = None
    
    def collect_system_metrics(self) -> List[Metric]:
        """Collect system-level metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(Metric(
            name="system_cpu_usage_percent",
            value=cpu_percent,
            timestamp=timestamp
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(Metric(
            name="system_memory_usage_percent",
            value=memory.percent,
            timestamp=timestamp
        ))
        metrics.append(Metric(
            name="system_memory_total_bytes",
            value=memory.total,
            timestamp=timestamp
        ))
        metrics.append(Metric(
            name="system_memory_available_bytes",
            value=memory.available,
            timestamp=timestamp
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(Metric(
            name="system_disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            timestamp=timestamp
        ))
        metrics.append(Metric(
            name="system_disk_total_bytes",
            value=disk.total,
            timestamp=timestamp
        ))
        metrics.append(Metric(
            name="system_disk_free_bytes",
            value=disk.free,
            timestamp=timestamp
        ))
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics.append(Metric(
            name="system_network_bytes_sent",
            value=net_io.bytes_sent,
            timestamp=timestamp
        ))
        metrics.append(Metric(
            name="system_network_bytes_recv",
            value=net_io.bytes_recv,
            timestamp=timestamp
        ))
        
        # Load average (Unix only)
        try:
            load1, load5, load15 = psutil.getloadavg()
            metrics.append(Metric(
                name="system_load_1m",
                value=load1,
                timestamp=timestamp
            ))
            metrics.append(Metric(
                name="system_load_5m",
                value=load5,
                timestamp=timestamp
            ))
            metrics.append(Metric(
                name="system_load_15m",
                value=load15,
                timestamp=timestamp
            ))
        except (AttributeError, OSError):
            pass  # Not available on Windows
        
        return metrics
    
    def collect_docker_metrics(self) -> List[Metric]:
        """Collect Docker container metrics."""
        if not self.docker_client:
            return []
        
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            containers = self.docker_client.containers.list()
            
            # Total container count
            metrics.append(Metric(
                name="docker_containers_total",
                value=len(containers),
                timestamp=timestamp
            ))
            
            # Running containers
            running_containers = [c for c in containers if c.status == 'running']
            metrics.append(Metric(
                name="docker_containers_running",
                value=len(running_containers),
                timestamp=timestamp
            ))
            
            # Per-container metrics
            for container in running_containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                     stats['precpu_stats']['system_cpu_usage']
                    
                    if system_cpu_delta > 0:
                        cpu_usage_percent = (cpu_delta / system_cpu_delta) * 100.0
                        metrics.append(Metric(
                            name="docker_container_cpu_usage_percent",
                            value=cpu_usage_percent,
                            labels={"container_name": container.name, "container_id": container.id[:12]},
                            timestamp=timestamp
                        ))
                    
                    # Memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    
                    if memory_limit > 0:
                        memory_usage_percent = (memory_usage / memory_limit) * 100.0
                        metrics.append(Metric(
                            name="docker_container_memory_usage_percent",
                            value=memory_usage_percent,
                            labels={"container_name": container.name, "container_id": container.id[:12]},
                            timestamp=timestamp
                        ))
                    
                    metrics.append(Metric(
                        name="docker_container_memory_usage_bytes",
                        value=memory_usage,
                        labels={"container_name": container.name, "container_id": container.id[:12]},
                        timestamp=timestamp
                    ))
                
                except Exception as e:
                    logger.warning(f"Failed to collect stats for container {container.name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to collect Docker metrics: {e}")
        
        return metrics
    
    def collect_process_metrics(self) -> List[Metric]:
        """Collect process-level metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Process count
        process_count = len(psutil.pids())
        metrics.append(Metric(
            name="system_processes_total",
            value=process_count,
            timestamp=timestamp
        ))
        
        # Get top processes by CPU and memory
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                pinfo = proc.info
                if pinfo['cpu_percent'] > 0:
                    processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage and get top 5
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        for i, proc in enumerate(processes[:5]):
            metrics.append(Metric(
                name="process_cpu_usage_percent",
                value=proc['cpu_percent'],
                labels={"process_name": proc['name'], "rank": str(i+1)},
                timestamp=timestamp
            ))
        
        return metrics
    
    def update_infrastructure_nodes(self):
        """Update infrastructure node information."""
        db = self.SessionLocal()
        try:
            # Get current node info
            import platform
            import socket
            
            hostname = socket.gethostname()
            
            # Check if node exists
            node = db.query(InfrastructureNode).filter(
                InfrastructureNode.hostname == hostname
            ).first()
            
            if not node:
                # Create new node
                node = InfrastructureNode(
                    name=hostname,
                    type="server",
                    hostname=hostname,
                    status="active",
                    os_info={
                        "system": platform.system(),
                        "release": platform.release(),
                        "version": platform.version(),
                        "machine": platform.machine(),
                        "processor": platform.processor()
                    }
                )
                db.add(node)
            
            # Update node information
            node.last_seen = datetime.utcnow()
            node.status = "active"
            
            # Update system specs
            node.cpu_cores = psutil.cpu_count()
            node.memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
            node.disk_gb = round(psutil.disk_usage('/').total / (1024**3), 2)
            
            # Try to get IP address
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                node.ip_address = s.getsockname()[0]
                s.close()
            except Exception:
                pass
            
            db.commit()
            logger.debug(f"Updated infrastructure node: {hostname}")
        
        except Exception as e:
            logger.error(f"Failed to update infrastructure node: {e}")
            db.rollback()
        finally:
            db.close()
    
    def cleanup_old_metrics(self):
        """Remove old metric records."""
        db = self.SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=settings.metrics_retention_days)
            
            deleted_count = db.query(Metric).filter(
                Metric.timestamp < cutoff_date
            ).delete()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old metric records")
                db.commit()
        
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def collect_and_store_metrics(self):
        """Collect all metrics and store in database."""
        logger.debug("Collecting metrics...")
        
        try:
            # Collect all metrics
            all_metrics = []
            all_metrics.extend(self.collect_system_metrics())
            all_metrics.extend(self.collect_docker_metrics())
            all_metrics.extend(self.collect_process_metrics())
            
            # Store metrics in database
            db = self.SessionLocal()
            try:
                for metric in all_metrics:
                    db.add(metric)
                
                db.commit()
                logger.debug(f"Stored {len(all_metrics)} metrics")
            
            except Exception as e:
                logger.error(f"Failed to store metrics: {e}")
                db.rollback()
            finally:
                db.close()
            
            # Update infrastructure nodes
            self.update_infrastructure_nodes()
        
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
    
    async def run(self):
        """Main metrics collection loop."""
        logger.info("Starting metrics collector")
        
        try:
            while True:
                # Collect metrics
                await self.collect_and_store_metrics()
                
                # Cleanup old metrics every hour
                if datetime.utcnow().minute == 0:
                    self.cleanup_old_metrics()
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect every minute
        
        except KeyboardInterrupt:
            logger.info("Metrics collector stopped")

async def main():
    """Run metrics collector."""
    collector = MetricsCollector()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Phase 5: Testing Strategy

### Comprehensive Test Suite

Create `tests/conftest.py`:

```python
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.platform.models.database import Base
from src.platform.database import get_db
from src.platform.api.main import app

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def db_session():
    """Create test database session."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up tables
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def override_get_db(db_session):
    """Override database dependency."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def client(override_get_db):
    """Create test client."""
    return TestClient(app)

@pytest.fixture
async def async_client(override_get_db):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
```

Create `tests/integration/test_full_workflow.py`:

```python
import pytest
import asyncio
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_complete_service_lifecycle(async_client: AsyncClient):
    """Test complete service management workflow."""
    
    # 1. Register a service
    service_data = {
        "name": "test-api",
        "url": "http://test-api.example.com",
        "environment": "test",
        "version": "1.0.0",
        "tags": ["api", "test"]
    }
    
    response = await async_client.post("/services/", json=service_data)
    assert response.status_code == 201
    service = response.json()
    service_id = service["id"]
    
    # 2. Verify service is listed
    response = await async_client.get("/services/")
    assert response.status_code == 200
    services = response.json()
    assert len(services) == 1
    assert services[0]["name"] == "test-api"
    
    # 3. Send heartbeat
    response = await async_client.post(f"/services/{service_id}/heartbeat")
    assert response.status_code == 200
    
    # 4. Get monitoring dashboard
    response = await async_client.get("/monitoring/dashboard")
    assert response.status_code == 200
    dashboard = response.json()
    assert dashboard["service_stats"]["total"] == 1
    
    # 5. Get health summary
    response = await async_client.get("/monitoring/health-summary")
    assert response.status_code == 200
    health_summary = response.json()
    assert health_summary["total_services"] == 1
    
    # 6. Update service
    update_data = {"version": "1.1.0", "tags": ["api", "test", "updated"]}
    response = await async_client.put(f"/services/{service_id}", json=update_data)
    assert response.status_code == 200
    updated_service = response.json()
    assert updated_service["version"] == "1.1.0"
    assert "updated" in updated_service["tags"]
    
    # 7. Delete service
    response = await async_client.delete(f"/services/{service_id}")
    assert response.status_code == 204
    
    # 8. Verify service is gone
    response = await async_client.get(f"/services/{service_id}")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_concurrent_service_operations(async_client: AsyncClient):
    """Test concurrent service operations."""
    
    # Create multiple services concurrently
    tasks = []
    for i in range(10):
        service_data = {
            "name": f"service-{i}",
            "url": f"http://service-{i}.example.com",
            "environment": "test"
        }
        tasks.append(async_client.post("/services/", json=service_data))
    
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    for response in responses:
        assert response.status_code == 201
    
    # Verify all services exist
    response = await async_client.get("/services/")
    assert response.status_code == 200
    services = response.json()
    assert len(services) == 10
    
    # Test concurrent heartbeats
    heartbeat_tasks = []
    for service in services:
        heartbeat_tasks.append(
            async_client.post(f"/services/{service['id']}/heartbeat")
        )
    
    heartbeat_responses = await asyncio.gather(*heartbeat_tasks)
    
    # All heartbeats should succeed
    for response in heartbeat_responses:
        assert response.status_code == 200
```

## Phase 6: Deployment and Final Integration

### Docker Configuration

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ src/
COPY config/ config/

# Create necessary directories
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "uvicorn", "src.platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://devops:password@db:5432/devops_platform
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=devops_platform
      - POSTGRES_USER=devops
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    restart: unless-stopped

  health_worker:
    build: .
    command: ["uv", "run", "python", "-m", "src.platform.workers.health_checker"]
    environment:
      - DATABASE_URL=postgresql://devops:password@db:5432/devops_platform
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  metrics_worker:
    build: .
    command: ["uv", "run", "python", "-m", "src.platform.workers.metrics_collector"]
    environment:
      - DATABASE_URL=postgresql://devops:password@db:5432/devops_platform
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/var/lib/grafana/dashboards
    restart: unless-stopped

volumes:
  postgres_data:
  grafana_data:
```

### Kubernetes Deployment

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: devops-platform
```

Create `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: devops-platform-config
  namespace: devops-platform
data:
  DATABASE_URL: "postgresql://devops:password@postgres:5432/devops_platform"
  REDIS_URL: "redis://redis:6379/0"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

### Final Project Tasks

1. **Complete Implementation:**
   - Implement all missing API endpoints
   - Add comprehensive error handling
   - Implement authentication and authorization
   - Add rate limiting and security measures

2. **Testing:**
   - Write unit tests for all components
   - Add integration tests for workflows
   - Implement performance testing
   - Add security testing

3. **Documentation:**
   - API documentation with OpenAPI
   - Deployment guides
   - User manual
   - Architecture documentation

4. **Monitoring:**
   - Set up Prometheus metrics
   - Create Grafana dashboards
   - Implement alerting rules
   - Add distributed tracing

5. **Production Readiness:**
   - Environment configuration
   - Secrets management
   - Backup strategies
   - Disaster recovery

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create course outline and structure", "status": "completed", "activeForm": "Creating course outline and structure"}, {"content": "Write Chapter 1: Python fundamentals for SRE/DevOps", "status": "completed", "activeForm": "Writing Chapter 1: Python fundamentals for SRE/DevOps"}, {"content": "Write Chapter 2: Virtual environments with uv", "status": "completed", "activeForm": "Writing Chapter 2: Virtual environments with uv"}, {"content": "Write Chapter 3: File operations and system administration", "status": "completed", "activeForm": "Writing Chapter 3: File operations and system administration"}, {"content": "Write Chapter 4: Working with APIs and HTTP requests", "status": "completed", "activeForm": "Writing Chapter 4: Working with APIs and HTTP requests"}, {"content": "Write Chapter 5: Building APIs with FastAPI", "status": "completed", "activeForm": "Writing Chapter 5: Building APIs with FastAPI"}, {"content": "Write Chapter 6: Error handling and logging", "status": "completed", "activeForm": "Writing Chapter 6: Error handling and logging"}, {"content": "Write Chapter 7: Testing and automation", "status": "completed", "activeForm": "Writing Chapter 7: Testing and automation"}, {"content": "Create practical exercises and projects", "status": "completed", "activeForm": "Creating practical exercises and projects"}]