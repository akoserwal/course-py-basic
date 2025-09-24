#!/usr/bin/env python3
"""
Comprehensive Monitoring and Alerting System
Advanced error handling, logging, and monitoring for DevOps applications.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import logging.handlers
from contextlib import contextmanager
import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

import httpx
import psutil
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Enums and Constants
class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class MonitoringMetric(str, Enum):
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"

# Data Models
@dataclass
class Alert:
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    service_name: Optional[str]
    metric_name: Optional[str]
    threshold_value: Optional[float]
    current_value: Optional[float]
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MetricThreshold:
    metric: MonitoringMetric
    warning_threshold: float
    critical_threshold: float
    comparison: str = "greater"  # greater, less, equal
    window_seconds: int = 300  # 5 minutes

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: Optional[float] = None
    active_connections: int = 0

class AlertRequest(BaseModel):
    title: str
    description: str
    severity: AlertSeverity
    service_name: Optional[str] = None
    metric_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MonitoringConfig(BaseModel):
    collection_interval: int = 60
    retention_days: int = 7
    alert_cooldown_minutes: int = 15
    thresholds: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Notification settings
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = Field(default_factory=list)

# Custom Exception Classes
class MonitoringError(Exception):
    """Base exception for monitoring system."""
    pass

class AlertingError(MonitoringError):
    """Exception raised when alerting fails."""
    pass

class MetricsCollectionError(MonitoringError):
    """Exception raised when metrics collection fails."""
    pass

# Enhanced Logging Setup
class StructuredLogger:
    """Structured logging with correlation IDs and contextual information."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)
        
        self.correlation_id = None
    
    def _setup_handlers(self, log_file: Optional[str] = None):
        """Set up logging handlers."""
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s]: %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Syslog handler (Unix only)
        try:
            if os.name != 'nt':  # Not Windows
                syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
                syslog_handler.setLevel(logging.WARNING)
                syslog_handler.setFormatter(formatter)
                self.logger.addHandler(syslog_handler)
        except Exception:
            pass  # Syslog not available
    
    @contextmanager
    def correlation_context(self, correlation_id: str):
        """Context manager for correlation ID."""
        old_correlation_id = self.correlation_id
        self.correlation_id = correlation_id
        try:
            yield
        finally:
            self.correlation_id = old_correlation_id
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with correlation ID and extra context."""
        extra = {
            'correlation_id': self.correlation_id or 'N/A',
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

# Retry Decorator with Exponential Backoff
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        await asyncio.sleep(delay)
                    else:
                        raise last_exception
        
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        time.sleep(delay)
                    else:
                        raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Metrics Collector
class MetricsCollector:
    """Collect system and application metrics."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.metrics_history: List[SystemMetrics] = []
        self.last_network_stats = None
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Load average (Unix only)
            load_average = None
            try:
                load_average = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                pass
            
            # Active network connections
            active_connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                active_connections=active_connections
            )
            
            self.metrics_history.append(metrics)
            
            # Limit history size
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            self.logger.debug(
                f"Collected metrics: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, Disk={disk_percent:.1f}%"
            )
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            raise MetricsCollectionError(f"Metrics collection failed: {e}")
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, minutes: int = 5) -> Optional[SystemMetrics]:
        """Get average metrics over specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        avg_connections = sum(m.active_connections for m in recent_metrics) / len(recent_metrics)
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            disk_percent=avg_disk,
            network_bytes_sent=recent_metrics[-1].network_bytes_sent,
            network_bytes_recv=recent_metrics[-1].network_bytes_recv,
            load_average=recent_metrics[-1].load_average,
            active_connections=int(avg_connections)
        )

# Alert Manager
class AlertManager:
    """Manage alerts with deduplication and notifications."""
    
    def __init__(self, config: MonitoringConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.alerts: Dict[str, Alert] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.notification_client = httpx.AsyncClient(timeout=30.0)
    
    def generate_alert_id(self, title: str, service_name: Optional[str] = None) -> str:
        """Generate unique alert ID."""
        import hashlib
        unique_string = f"{title}:{service_name}:{datetime.utcnow().date()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def create_alert(self, alert_request: AlertRequest) -> Alert:
        """Create a new alert."""
        alert_id = self.generate_alert_id(alert_request.title, alert_request.service_name)
        
        # Check for cooldown
        if alert_id in self.alert_cooldowns:
            cooldown_until = self.alert_cooldowns[alert_id] + timedelta(
                minutes=self.config.alert_cooldown_minutes
            )
            if datetime.utcnow() < cooldown_until:
                self.logger.debug(f"Alert {alert_id} is in cooldown")
                return self.alerts.get(alert_id)
        
        # Check if alert already exists and is active
        if alert_id in self.alerts and self.alerts[alert_id].status == AlertStatus.ACTIVE:
            self.logger.debug(f"Alert {alert_id} already exists and is active")
            return self.alerts[alert_id]
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            title=alert_request.title,
            description=alert_request.description,
            severity=alert_request.severity,
            status=AlertStatus.ACTIVE,
            service_name=alert_request.service_name,
            metric_name=alert_request.metric_name,
            threshold_value=None,
            current_value=None,
            created_at=datetime.utcnow(),
            metadata=alert_request.metadata
        )
        
        self.alerts[alert_id] = alert
        self.alert_cooldowns[alert_id] = datetime.utcnow()
        
        self.logger.info(f"Created alert: {alert.title} (ID: {alert_id})")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            
            self.logger.info(f"Acknowledged alert: {alert_id}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            self.logger.info(f"Resolved alert: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    @retry_with_backoff(max_retries=3)
    async def send_slack_notification(self, alert: Alert):
        """Send alert notification to Slack."""
        if not self.config.slack_webhook_url:
            return
        
        color_map = {
            AlertSeverity.LOW: "#36a64f",
            AlertSeverity.MEDIUM: "#ff9900",
            AlertSeverity.HIGH: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"🚨 {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.upper(),
                            "short": True
                        },
                        {
                            "title": "Service",
                            "value": alert.service_name or "N/A",
                            "short": True
                        },
                        {
                            "title": "Created",
                            "value": alert.created_at.isoformat(),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        try:
            response = await self.notification_client.post(
                self.config.slack_webhook_url,
                json=payload
            )
            response.raise_for_status()
            self.logger.info(f"Sent Slack notification for alert {alert.id}")
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            raise AlertingError(f"Slack notification failed: {e}")
    
    @retry_with_backoff(max_retries=3)
    async def send_email_notification(self, alert: Alert):
        """Send alert notification via email."""
        if not all([
            self.config.email_smtp_server,
            self.config.email_username,
            self.config.email_password,
            self.config.email_recipients
        ]):
            return
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"Alert: {alert.title} ({alert.severity.upper()})"
            
            # Email body
            body = f"""
Alert Details:
--------------
Title: {alert.title}
Description: {alert.description}
Severity: {alert.severity.upper()}
Service: {alert.service_name or 'N/A'}
Created: {alert.created_at.isoformat()}
Alert ID: {alert.id}

Status: {alert.status.upper()}
Metadata: {json.dumps(alert.metadata, indent=2)}
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Sent email notification for alert {alert.id}")
        
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            raise AlertingError(f"Email notification failed: {e}")
    
    async def send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        notification_tasks = []
        
        if self.config.slack_webhook_url:
            notification_tasks.append(self.send_slack_notification(alert))
        
        if self.config.email_recipients:
            notification_tasks.append(self.send_email_notification(alert))
        
        if notification_tasks:
            try:
                await asyncio.gather(*notification_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"Some notifications failed: {e}")

# Threshold Monitor
class ThresholdMonitor:
    """Monitor metrics against defined thresholds."""
    
    def __init__(self, alert_manager: AlertManager, logger: StructuredLogger):
        self.alert_manager = alert_manager
        self.logger = logger
        self.thresholds: List[MetricThreshold] = []
    
    def add_threshold(self, threshold: MetricThreshold):
        """Add a metric threshold."""
        self.thresholds.append(threshold)
        self.logger.info(f"Added threshold for {threshold.metric}: {threshold.warning_threshold}/{threshold.critical_threshold}")
    
    def check_thresholds(self, metrics: SystemMetrics):
        """Check current metrics against thresholds."""
        for threshold in self.thresholds:
            current_value = self._get_metric_value(metrics, threshold.metric)
            
            if current_value is None:
                continue
            
            severity = self._evaluate_threshold(current_value, threshold)
            
            if severity:
                # Create alert
                alert_request = AlertRequest(
                    title=f"{threshold.metric.value.replace('_', ' ').title()} Threshold Exceeded",
                    description=f"{threshold.metric.value} is {current_value:.2f}, exceeding {severity} threshold",
                    severity=severity,
                    metric_name=threshold.metric.value,
                    metadata={
                        "current_value": current_value,
                        "threshold_warning": threshold.warning_threshold,
                        "threshold_critical": threshold.critical_threshold,
                        "comparison": threshold.comparison
                    }
                )
                
                alert = self.alert_manager.create_alert(alert_request)
                
                # Send notifications asynchronously
                asyncio.create_task(self.alert_manager.send_notifications(alert))
    
    def _get_metric_value(self, metrics: SystemMetrics, metric: MonitoringMetric) -> Optional[float]:
        """Get metric value from SystemMetrics object."""
        metric_map = {
            MonitoringMetric.CPU_USAGE: metrics.cpu_percent,
            MonitoringMetric.MEMORY_USAGE: metrics.memory_percent,
            MonitoringMetric.DISK_USAGE: metrics.disk_percent,
            MonitoringMetric.NETWORK_IO: max(metrics.network_bytes_sent, metrics.network_bytes_recv) / (1024 * 1024),  # MB
        }
        
        return metric_map.get(metric)
    
    def _evaluate_threshold(self, current_value: float, threshold: MetricThreshold) -> Optional[AlertSeverity]:
        """Evaluate if current value exceeds threshold."""
        if threshold.comparison == "greater":
            if current_value >= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif current_value >= threshold.warning_threshold:
                return AlertSeverity.HIGH
        elif threshold.comparison == "less":
            if current_value <= threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            elif current_value <= threshold.warning_threshold:
                return AlertSeverity.HIGH
        
        return None

# FastAPI Application
app = FastAPI(
    title="Comprehensive Monitoring System",
    description="Advanced monitoring, alerting, and error handling system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
logger = StructuredLogger("monitoring_system", "monitoring.log")
config = MonitoringConfig()
metrics_collector = MetricsCollector(logger)
alert_manager = AlertManager(config, logger)
threshold_monitor = ThresholdMonitor(alert_manager, logger)

# Add default thresholds
threshold_monitor.add_threshold(MetricThreshold(
    metric=MonitoringMetric.CPU_USAGE,
    warning_threshold=80.0,
    critical_threshold=95.0
))

threshold_monitor.add_threshold(MetricThreshold(
    metric=MonitoringMetric.MEMORY_USAGE,
    warning_threshold=85.0,
    critical_threshold=95.0
))

threshold_monitor.add_threshold(MetricThreshold(
    metric=MonitoringMetric.DISK_USAGE,
    warning_threshold=80.0,
    critical_threshold=90.0
))

# Background Tasks
async def metrics_collection_worker():
    """Background worker for metrics collection."""
    while True:
        try:
            with logger.correlation_context("metrics_worker"):
                metrics = metrics_collector.collect_system_metrics()
                threshold_monitor.check_thresholds(metrics)
                
                logger.debug("Completed metrics collection cycle")
        except Exception as e:
            logger.error(f"Metrics collection worker error: {e}")
            logger.error(traceback.format_exc())
        
        await asyncio.sleep(config.collection_interval)

@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    logger.info("Starting monitoring system")
    asyncio.create_task(metrics_collection_worker())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down monitoring system")
    await alert_manager.notification_client.aclose()

# Request logging middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all HTTP requests."""
    correlation_id = request.headers.get("X-Correlation-ID", f"req_{int(time.time()*1000)}")
    
    with logger.correlation_context(correlation_id):
        start_time = time.time()
        
        logger.info(f"Request started: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            duration = (time.time() - start_time) * 1000
            
            logger.info(f"Request completed: {response.status_code} in {duration:.2f}ms")
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Request failed: {e} after {duration:.2f}ms")
            raise

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Comprehensive Monitoring System",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check system resources
        metrics = metrics_collector.collect_system_metrics()
        active_alerts = alert_manager.get_active_alerts()
        
        status = "healthy"
        if len(active_alerts) > 0:
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            if critical_alerts:
                status = "critical"
            else:
                status = "warning"
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
                "active_connections": metrics.active_connections
            },
            "alerts": {
                "total": len(alert_manager.alerts),
                "active": len(active_alerts),
                "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics/current")
async def get_current_metrics():
    """Get current system metrics."""
    try:
        metrics = metrics_collector.collect_system_metrics()
        return asdict(metrics)
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/metrics/history")
async def get_metrics_history(hours: int = 24):
    """Get metrics history."""
    try:
        history = metrics_collector.get_metrics_history(hours)
        return [asdict(m) for m in history]
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics history")

@app.post("/alerts")
async def create_alert(alert_request: AlertRequest, background_tasks: BackgroundTasks):
    """Create a new alert."""
    try:
        alert = alert_manager.create_alert(alert_request)
        
        # Send notifications in background
        background_tasks.add_task(alert_manager.send_notifications, alert)
        
        return asdict(alert)
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert")

@app.get("/alerts")
async def list_alerts(status_filter: Optional[AlertStatus] = None, severity_filter: Optional[AlertSeverity] = None):
    """List alerts with optional filtering."""
    try:
        alerts = list(alert_manager.alerts.values())
        
        if status_filter:
            alerts = [a for a in alerts if a.status == status_filter]
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        return [asdict(alert) for alert in alerts]
    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        success = alert_manager.acknowledge_alert(alert_id)
        if success:
            return {"message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    try:
        success = alert_manager.resolve_alert(alert_id)
        if success:
            return {"message": f"Alert {alert_id} resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")

@app.get("/stats/overview")
async def get_overview_stats():
    """Get system overview statistics."""
    try:
        current_metrics = metrics_collector.collect_system_metrics()
        active_alerts = alert_manager.get_active_alerts()
        
        # Calculate alert statistics
        alert_stats = {
            "total": len(alert_manager.alerts),
            "active": len(active_alerts),
            "by_severity": {}
        }
        
        for severity in AlertSeverity:
            alert_stats["by_severity"][severity.value] = len([
                a for a in active_alerts if a.severity == severity
            ])
        
        return {
            "system_metrics": asdict(current_metrics),
            "alert_statistics": alert_stats,
            "monitoring_config": {
                "collection_interval": config.collection_interval,
                "retention_days": config.retention_days,
                "alert_cooldown_minutes": config.alert_cooldown_minutes
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get overview stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve overview statistics")

if __name__ == "__main__":
    # Load configuration from environment
    config.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    config.email_smtp_server = os.getenv("EMAIL_SMTP_SERVER")
    config.email_username = os.getenv("EMAIL_USERNAME")
    config.email_password = os.getenv("EMAIL_PASSWORD")
    
    if os.getenv("EMAIL_RECIPIENTS"):
        config.email_recipients = os.getenv("EMAIL_RECIPIENTS").split(",")
    
    uvicorn.run(
        "comprehensive_monitoring:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )