# Chapter 6: Error Handling and Logging

## Learning Objectives
- Master Python exception handling for robust applications
- Implement comprehensive logging strategies for DevOps
- Create structured logging for monitoring and alerting systems
- Build error recovery and resilience mechanisms
- Integrate logging with monitoring tools and log aggregation systems
- Design alerting based on application logs and metrics

## 6.1 Exception Handling Best Practices

### Understanding Python Exceptions

```python
import sys
import traceback
from typing import Optional, Dict, Any
from datetime import datetime

class DevOpsError(Exception):
    """Base exception for DevOps-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

class ServiceUnavailableError(DevOpsError):
    """Exception for service unavailability."""
    pass

class ConfigurationError(DevOpsError):
    """Exception for configuration issues."""
    pass

class DeploymentError(DevOpsError):
    """Exception for deployment failures."""
    pass

class MonitoringError(DevOpsError):
    """Exception for monitoring system issues."""
    pass

def handle_service_call(service_url: str, timeout: int = 30):
    """Example of proper exception handling for service calls."""
    import requests
    
    try:
        response = requests.get(service_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        raise ServiceUnavailableError(
            f"Service at {service_url} timed out",
            details={'timeout': timeout, 'service_url': service_url}
        )
    
    except requests.exceptions.ConnectionError:
        raise ServiceUnavailableError(
            f"Cannot connect to service at {service_url}",
            details={'service_url': service_url}
        )
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code >= 500:
            raise ServiceUnavailableError(
                f"Service error: {e.response.status_code}",
                details={'status_code': e.response.status_code, 'service_url': service_url}
            )
        else:
            raise DevOpsError(
                f"Client error: {e.response.status_code}",
                details={'status_code': e.response.status_code, 'service_url': service_url}
            )
    
    except ValueError as e:
        raise DevOpsError(
            "Invalid JSON response from service",
            details={'service_url': service_url, 'error': str(e)}
        )
```

### Exception Context Management

```python
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def error_context(operation: str, **context):
    """Context manager for consistent error handling."""
    try:
        logger.info(f"Starting operation: {operation}", extra=context)
        yield
        logger.info(f"Completed operation: {operation}", extra=context)
    
    except DevOpsError as e:
        logger.error(
            f"DevOps error in {operation}: {e.message}",
            extra={**context, **e.details, 'error_type': type(e).__name__}
        )
        raise
    
    except Exception as e:
        logger.error(
            f"Unexpected error in {operation}: {str(e)}",
            extra={**context, 'error_type': type(e).__name__},
            exc_info=True
        )
        raise DevOpsError(
            f"Unexpected error in {operation}",
            details={**context, 'original_error': str(e)}
        )

# Usage example
def deploy_service(service_name: str, version: str, environment: str):
    """Deploy service with proper error handling."""
    with error_context("deploy_service", 
                      service_name=service_name, 
                      version=version, 
                      environment=environment):
        
        # Validate configuration
        if not service_name or not version:
            raise ConfigurationError("Service name and version are required")
        
        # Check environment
        valid_environments = ['dev', 'staging', 'prod']
        if environment not in valid_environments:
            raise ConfigurationError(
                f"Invalid environment: {environment}",
                details={'valid_environments': valid_environments}
            )
        
        # Simulate deployment steps
        _validate_deployment_config(service_name, version, environment)
        _build_service(service_name, version)
        _deploy_to_environment(service_name, version, environment)
        _verify_deployment(service_name, environment)

def _validate_deployment_config(service_name: str, version: str, environment: str):
    """Validate deployment configuration."""
    # Simulate configuration validation
    pass

def _build_service(service_name: str, version: str):
    """Build service artifacts."""
    # Simulate build process
    pass

def _deploy_to_environment(service_name: str, version: str, environment: str):
    """Deploy service to target environment."""
    # Simulate deployment
    pass

def _verify_deployment(service_name: str, environment: str):
    """Verify deployment was successful."""
    # Simulate deployment verification
    pass
```

### Retry Logic with Exponential Backoff

```python
import time
import random
from functools import wraps
from typing import Callable, Type, Tuple

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for retrying functions with exponential backoff."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            extra={'function': func.__name__, 'attempts': attempt + 1},
                            exc_info=True
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {delay:.2f}s",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'delay': delay,
                            'error': str(e)
                        }
                    )
                    
                    time.sleep(delay)
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator

# Example usage
@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ServiceUnavailableError, requests.exceptions.RequestException)
)
def check_service_health(service_url: str) -> Dict[str, Any]:
    """Check service health with automatic retries."""
    return handle_service_call(f"{service_url}/health")
```

## 6.2 Python Logging Module Deep Dive

### Logging Configuration

```python
import logging
import logging.config
import json
import sys
from datetime import datetime
from pathlib import Path

# Logging configuration dictionary
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/devops.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10
        },
        'syslog': {
            'class': 'logging.handlers.SysLogHandler',
            'level': 'WARNING',
            'formatter': 'standard',
            'address': '/dev/log'  # Unix socket for syslog
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'devops': {
            'handlers': ['console', 'file', 'error_file', 'syslog'],
            'level': 'DEBUG',
            'propagate': False
        },
        'requests': {
            'level': 'WARNING',
            'propagate': True
        },
        'urllib3': {
            'level': 'WARNING',
            'propagate': True
        }
    }
}

def setup_logging(config_dict: dict = None, log_dir: str = "logs"):
    """Setup logging configuration."""
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Use provided config or default
    config = config_dict or LOGGING_CONFIG
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger('devops')
    logger.info("Logging system initialized")
```

### Structured Logging

```python
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

class StructuredLogger:
    """Structured logger for DevOps operations."""
    
    def __init__(self, name: str, correlation_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.context = {
            'service': name,
            'correlation_id': self.correlation_id
        }
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal log method with structured data."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level,
            'message': message,
            **self.context,
            **kwargs
        }
        
        # Use appropriate log level
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with optional exception details."""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            })
        self._log('ERROR', message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log('DEBUG', message, **kwargs)
    
    def with_context(self, **context) -> 'StructuredLogger':
        """Create new logger with additional context."""
        new_logger = StructuredLogger(self.logger.name, self.correlation_id)
        new_logger.context.update(context)
        return new_logger

# Usage examples
def process_deployment_with_logging(service_name: str, version: str):
    """Example of structured logging in deployment process."""
    logger = StructuredLogger('deployment').with_context(
        service_name=service_name,
        version=version,
        operation='deploy'
    )
    
    logger.info("Starting deployment", environment='production')
    
    try:
        # Simulate deployment steps
        logger.info("Validating configuration")
        _validate_config(service_name, version)
        
        logger.info("Building artifacts")
        _build_artifacts(logger, service_name, version)
        
        logger.info("Deploying to production")
        _deploy_to_production(logger, service_name, version)
        
        logger.info("Deployment completed successfully", 
                   duration_seconds=120, 
                   status='success')
    
    except Exception as e:
        logger.error("Deployment failed", error=e, status='failed')
        raise

def _validate_config(service_name: str, version: str):
    """Validate deployment configuration."""
    # Simulation
    pass

def _build_artifacts(logger: StructuredLogger, service_name: str, version: str):
    """Build deployment artifacts."""
    build_logger = logger.with_context(step='build')
    build_logger.info("Starting build process")
    
    # Simulate build steps
    build_logger.debug("Downloading dependencies")
    build_logger.debug("Compiling application")
    build_logger.debug("Running tests")
    
    build_logger.info("Build completed", artifact_size_mb=150)

def _deploy_to_production(logger: StructuredLogger, service_name: str, version: str):
    """Deploy to production environment."""
    deploy_logger = logger.with_context(step='deploy', environment='production')
    deploy_logger.info("Starting production deployment")
    
    # Simulate deployment
    deploy_logger.info("Stopping old version")
    deploy_logger.info("Starting new version") 
    deploy_logger.info("Health check passed")
```

## 6.3 Application Metrics and Monitoring

### Custom Metrics Collection

```python
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

@dataclass
class MetricPoint:
    """Single metric data point."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points = max_points_per_metric
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_points))
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Add observation to histogram."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(MetricPoint(value, datetime.now(), labels or {}))
    
    def start_timer(self, name: str, labels: Dict[str, str] = None):
        """Start timing an operation."""
        return Timer(self, name, labels)
    
    def record_timing(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record timing measurement."""
        with self._lock:
            key = self._make_key(name, labels)
            self._timers[key].append(duration)
            
            # Keep only last 1000 timings
            if len(self._timers[key]) > 1000:
                self._timers[key] = self._timers[key][-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {
                    name: [{'value': p.value, 'timestamp': p.timestamp.isoformat()} for p in points]
                    for name, points in self._histograms.items()
                },
                'timers': {
                    name: {
                        'count': len(timings),
                        'average': sum(timings) / len(timings) if timings else 0,
                        'min': min(timings) if timings else 0,
                        'max': max(timings) if timings else 0
                    }
                    for name, timings in self._timers.items()
                }
            }
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f'{name}{{{label_str}}}'

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timing(self.name, duration, self.labels)

# Global metrics collector
metrics = MetricsCollector()

# Decorators for automatic metrics collection
def count_calls(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to count function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.increment_counter(metric_name, labels=labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def time_function(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with metrics.start_timer(metric_name, labels=labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
@count_calls('deployments_total', {'operation': 'deploy'})
@time_function('deployment_duration_seconds')
def deploy_service_with_metrics(service_name: str, version: str):
    """Deploy service with automatic metrics collection."""
    logger = StructuredLogger('deployment')
    
    try:
        logger.info("Starting deployment", service=service_name, version=version)
        
        # Simulate deployment work
        time.sleep(2)
        
        metrics.set_gauge('last_deployment_timestamp', time.time(), 
                         {'service': service_name})
        
        logger.info("Deployment successful")
        
    except Exception as e:
        metrics.increment_counter('deployment_errors_total', 
                                 labels={'service': service_name, 'error_type': type(e).__name__})
        logger.error("Deployment failed", error=e)
        raise
```

## 6.4 Alerting and Monitoring Integration

### Alert Manager

```python
import requests
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure."""
    title: str
    message: str
    severity: AlertSeverity
    service: str
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts: List[Alert] = []
        self.logger = StructuredLogger('alerting')
    
    def send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        self.logger.info(
            "Sending alert",
            title=alert.title,
            severity=alert.severity.value,
            service=alert.service
        )
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Send to configured channels
        if self.config.get('slack', {}).get('enabled'):
            self._send_slack_alert(alert)
        
        if self.config.get('email', {}).get('enabled'):
            self._send_email_alert(alert)
        
        if self.config.get('pagerduty', {}).get('enabled'):
            self._send_pagerduty_alert(alert)
        
        if self.config.get('webhook', {}).get('enabled'):
            self._send_webhook_alert(alert)
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        webhook_url = self.config['slack']['webhook_url']
        
        color_map = {
            AlertSeverity.LOW: "good",
            AlertSeverity.MEDIUM: "warning", 
            AlertSeverity.HIGH: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "text": f"🚨 {alert.severity.value.upper()} Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "warning"),
                    "fields": [
                        {
                            "title": "Service",
                            "value": alert.service,
                            "short": True
                        },
                        {
                            "title": "Severity", 
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Slack alert sent successfully")
        except Exception as e:
            self.logger.error("Failed to send Slack alert", error=e)
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        # Implementation would depend on email service (SMTP, SendGrid, etc.)
        self.logger.info("Email alert sent", recipient=self.config['email']['recipients'])
    
    def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty."""
        api_key = self.config['pagerduty']['api_key']
        
        payload = {
            "routing_key": api_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "source": alert.service,
                "severity": alert.severity.value,
                "component": alert.service,
                "custom_details": {
                    "message": alert.message,
                    "labels": alert.labels,
                    "annotations": alert.annotations
                }
            }
        }
        
        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            self.logger.info("PagerDuty alert sent successfully")
        except Exception as e:
            self.logger.error("Failed to send PagerDuty alert", error=e)
    
    def _send_webhook_alert(self, alert: Alert):
        """Send alert to custom webhook."""
        webhook_url = self.config['webhook']['url']
        
        payload = {
            "alert": {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "service": alert.service,
                "timestamp": alert.timestamp.isoformat(),
                "labels": alert.labels,
                "annotations": alert.annotations
            }
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            self.logger.info("Webhook alert sent successfully")
        except Exception as e:
            self.logger.error("Failed to send webhook alert", error=e)

# Alert rules and monitoring
class AlertRule:
    """Define alerting rules based on metrics."""
    
    def __init__(self, name: str, condition: callable, alert_config: Dict[str, Any]):
        self.name = name
        self.condition = condition
        self.alert_config = alert_config
        self.last_triggered = None
        self.cooldown_minutes = alert_config.get('cooldown_minutes', 5)
    
    def evaluate(self, metrics_data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate rule against metrics data."""
        if self.condition(metrics_data):
            # Check cooldown period
            if (self.last_triggered and 
                (datetime.now() - self.last_triggered).total_seconds() < self.cooldown_minutes * 60):
                return None
            
            self.last_triggered = datetime.now()
            
            return Alert(
                title=self.alert_config['title'],
                message=self.alert_config['message'],
                severity=AlertSeverity(self.alert_config['severity']),
                service=self.alert_config['service'],
                labels=self.alert_config.get('labels', {}),
                annotations=self.alert_config.get('annotations', {})
            )
        
        return None

class MonitoringSystem:
    """Complete monitoring and alerting system."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.alert_rules: List[AlertRule] = []
        self.logger = StructuredLogger('monitoring')
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alerting rule."""
        self.alert_rules.append(rule)
        self.logger.info("Added alert rule", rule_name=rule.name)
    
    def evaluate_alerts(self, metrics_data: Dict[str, Any]):
        """Evaluate all alert rules against metrics."""
        for rule in self.alert_rules:
            try:
                alert = rule.evaluate(metrics_data)
                if alert:
                    self.logger.warning(
                        "Alert triggered",
                        rule_name=rule.name,
                        alert_title=alert.title,
                        severity=alert.severity.value
                    )
                    self.alert_manager.send_alert(alert)
            
            except Exception as e:
                self.logger.error(
                    "Error evaluating alert rule",
                    rule_name=rule.name,
                    error=e
                )
```

## Exercise 6: Build a Comprehensive Monitoring and Alerting System

### Exercise Overview
Create a monitoring system that tracks application health, collects metrics, handles errors gracefully, and sends alerts when issues are detected.

### Step 1: Project Setup

```bash
mkdir monitoring-system
cd monitoring-system
uv init
uv add fastapi uvicorn requests pyyaml python-json-logger
mkdir -p {config,logs,tests}
```

### Step 2: Create Monitoring Application

Create `src/monitoring_system/app.py`:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .logging_config import setup_logging, StructuredLogger
from .metrics import MetricsCollector, Timer
from .alerting import AlertManager, Alert, AlertSeverity, AlertRule, MonitoringSystem

# Setup logging
setup_logging()

# Initialize components
app = FastAPI(title="Monitoring System", version="1.0.0")
metrics = MetricsCollector()
logger = StructuredLogger('monitoring-api')

# Configuration
ALERT_CONFIG = {
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    },
    'email': {
        'enabled': False,
        'recipients': ['ops@company.com']
    }
}

alert_manager = AlertManager(ALERT_CONFIG)
monitoring_system = MonitoringSystem(alert_manager)

# Add some example alert rules
def high_error_rate_condition(metrics_data: Dict[str, Any]) -> bool:
    """Check for high error rate."""
    counters = metrics_data.get('counters', {})
    total_requests = counters.get('http_requests_total', 0)
    error_requests = counters.get('http_errors_total', 0)
    
    if total_requests > 100:  # Only alert if we have enough data
        error_rate = (error_requests / total_requests) * 100
        return error_rate > 5  # Alert if error rate > 5%
    
    return False

def slow_response_time_condition(metrics_data: Dict[str, Any]) -> bool:
    """Check for slow response times."""
    timers = metrics_data.get('timers', {})
    response_times = timers.get('http_request_duration_seconds', {})
    avg_response_time = response_times.get('average', 0)
    
    return avg_response_time > 2.0  # Alert if average response time > 2 seconds

monitoring_system.add_alert_rule(AlertRule(
    name="high_error_rate",
    condition=high_error_rate_condition,
    alert_config={
        'title': 'High Error Rate Detected',
        'message': 'Application error rate exceeds 5%',
        'severity': 'high',
        'service': 'monitoring-api',
        'cooldown_minutes': 5
    }
))

monitoring_system.add_alert_rule(AlertRule(
    name="slow_response_time", 
    condition=slow_response_time_condition,
    alert_config={
        'title': 'Slow Response Time Detected',
        'message': 'Average response time exceeds 2 seconds',
        'severity': 'medium',
        'service': 'monitoring-api',
        'cooldown_minutes': 10
    }
))

# Middleware for request metrics and logging
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        "HTTP request received",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )
    
    # Increment request counter
    metrics.increment_counter('http_requests_total', labels={'method': request.method})
    
    try:
        response = await call_next(request)
        
        # Record successful request
        duration = time.time() - start_time
        metrics.record_timing('http_request_duration_seconds', duration, 
                             labels={'method': request.method, 'status': str(response.status_code)})
        
        # Log response
        logger.info(
            "HTTP request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
        
        return response
    
    except Exception as e:
        # Record error
        duration = time.time() - start_time
        metrics.increment_counter('http_errors_total', labels={'method': request.method})
        metrics.record_timing('http_request_duration_seconds', duration,
                             labels={'method': request.method, 'status': '500'})
        
        # Log error
        logger.error(
            "HTTP request failed",
            method=request.method,
            path=request.url.path,
            duration_seconds=duration,
            error=e
        )
        
        raise

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Monitoring System API", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus-style metrics endpoint."""
    metrics_data = metrics.get_metrics()
    
    lines = []
    
    # Export counters
    for name, value in metrics_data['counters'].items():
        lines.append(f"{name} {value}")
    
    # Export gauges
    for name, value in metrics_data['gauges'].items():
        lines.append(f"{name} {value}")
    
    # Export timer averages
    for name, timer_data in metrics_data['timers'].items():
        lines.append(f"{name}_avg {timer_data['average']}")
        lines.append(f"{name}_count {timer_data['count']}")
        lines.append(f"{name}_min {timer_data['min']}")
        lines.append(f"{name}_max {timer_data['max']}")
    
    return "\n".join(lines)

@app.get("/simulate/error")
async def simulate_error():
    """Simulate an error for testing."""
    logger.warning("Simulating error for testing")
    
    if random.random() < 0.8:  # 80% chance of error
        raise HTTPException(status_code=500, detail="Simulated error")
    
    return {"message": "No error this time!"}

@app.get("/simulate/slow")
async def simulate_slow_response():
    """Simulate slow response for testing."""
    delay = random.uniform(1, 5)  # 1-5 second delay
    logger.info("Simulating slow response", delay_seconds=delay)
    
    await asyncio.sleep(delay)
    
    return {"message": f"Delayed response after {delay:.2f} seconds"}

@app.post("/alerts/test")
async def send_test_alert():
    """Send a test alert."""
    test_alert = Alert(
        title="Test Alert",
        message="This is a test alert from the monitoring system",
        severity=AlertSeverity.LOW,
        service="monitoring-api"
    )
    
    alert_manager.send_alert(test_alert)
    
    return {"message": "Test alert sent"}

# Background task for monitoring
async def monitoring_loop():
    """Background task that runs monitoring checks."""
    while True:
        try:
            # Get current metrics
            metrics_data = metrics.get_metrics()
            
            # Evaluate alert rules
            monitoring_system.evaluate_alerts(metrics_data)
            
            # Update system metrics
            metrics.set_gauge('monitoring_last_run_timestamp', time.time())
            
            # Wait before next check
            await asyncio.sleep(30)  # Check every 30 seconds
        
        except Exception as e:
            logger.error("Error in monitoring loop", error=e)
            await asyncio.sleep(60)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    app.state.start_time = time.time()
    logger.info("Monitoring system starting up")
    
    # Start background monitoring
    asyncio.create_task(monitoring_loop())
    
    # Set initial metrics
    metrics.set_gauge('app_start_timestamp', app.state.start_time)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Monitoring system shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 3: Create Load Testing Script

Create `tests/load_test.py`:

```python
#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import random
from typing import List

async def make_request(session: aiohttp.ClientSession, url: str) -> dict:
    """Make a single HTTP request."""
    start_time = time.time()
    try:
        async with session.get(url) as response:
            duration = time.time() - start_time
            return {
                'url': url,
                'status': response.status,
                'duration': duration,
                'success': response.status < 400
            }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'url': url,
            'status': 0,
            'duration': duration,
            'success': False,
            'error': str(e)
        }

async def load_test(base_url: str = "http://localhost:8000", 
                   concurrent_requests: int = 10,
                   total_requests: int = 100,
                   error_rate: float = 0.2):
    """Run load test against the monitoring API."""
    
    urls = [
        f"{base_url}/",
        f"{base_url}/health",
        f"{base_url}/metrics",
    ]
    
    # Add error endpoints based on error rate
    if random.random() < error_rate:
        urls.extend([
            f"{base_url}/simulate/error",
            f"{base_url}/simulate/slow"
        ])
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request():
            async with semaphore:
                url = random.choice(urls)
                return await make_request(session, url)
        
        # Create tasks
        tasks = [bounded_request() for _ in range(total_requests)]
        
        # Execute requests
        print(f"Starting load test: {total_requests} requests with {concurrent_requests} concurrent")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = time.time() - start_time
    
    # Analyze results
    successful_requests = [r for r in results if isinstance(r, dict) and r['success']]
    failed_requests = [r for r in results if isinstance(r, dict) and not r['success']]
    
    if successful_requests:
        avg_response_time = sum(r['duration'] for r in successful_requests) / len(successful_requests)
        max_response_time = max(r['duration'] for r in successful_requests)
        min_response_time = min(r['duration'] for r in successful_requests)
    else:
        avg_response_time = max_response_time = min_response_time = 0
    
    print(f"\nLoad Test Results:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful: {len(successful_requests)}")
    print(f"Failed: {len(failed_requests)}")
    print(f"Success Rate: {len(successful_requests)/total_requests*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Requests/Second: {total_requests/total_duration:.2f}")
    print(f"Average Response Time: {avg_response_time:.3f}s")
    print(f"Min Response Time: {min_response_time:.3f}s")
    print(f"Max Response Time: {max_response_time:.3f}s")

if __name__ == "__main__":
    asyncio.run(load_test(
        concurrent_requests=20,
        total_requests=500,
        error_rate=0.3  # 30% of requests will hit error endpoints
    ))
```

### Step 4: Exercise Tasks

1. **Start the monitoring system:**
   ```bash
   uv run python -m monitoring_system.app
   ```

2. **View metrics:**
   ```bash
   curl http://localhost:8000/metrics
   ```

3. **Run load test to generate traffic:**
   ```bash
   uv run python tests/load_test.py
   ```

4. **Test error scenarios:**
   ```bash
   # Generate errors
   for i in {1..10}; do curl http://localhost:8000/simulate/error; done
   
   # Generate slow responses
   for i in {1..5}; do curl http://localhost:8000/simulate/slow; done
   ```

5. **Send test alert:**
   ```bash
   curl -X POST http://localhost:8000/alerts/test
   ```

6. **Advanced challenges:**
   - Integrate with Prometheus for metrics storage
   - Add custom dashboards with Grafana
   - Implement log aggregation with ELK stack
   - Create deployment health checks
   - Add circuit breaker patterns
   - Implement distributed tracing

## Key Takeaways

- Proper exception handling prevents cascading failures
- Structured logging enables effective troubleshooting and monitoring
- Metrics collection provides insights into application performance
- Automated alerting enables proactive incident response
- Retry logic with backoff handles transient failures gracefully
- Context managers ensure consistent error handling patterns
- Log aggregation and correlation IDs improve debugging in distributed systems

This comprehensive approach to error handling and logging forms the foundation for reliable, observable applications in production environments.