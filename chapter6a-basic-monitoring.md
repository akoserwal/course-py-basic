# Chapter 6A: Basic Monitoring and Logging

## Learning Objectives
- Understand why monitoring matters in DevOps
- Set up basic logging in Python applications
- Monitor system resources (CPU, memory, disk)
- Create simple alerts and notifications
- Handle errors gracefully

## 6A.1 Why Monitor Everything?

In DevOps, monitoring is like being a doctor for your systems:
- **Prevention**: Catch problems before they become disasters
- **Diagnosis**: Understand what went wrong when things break
- **Performance**: Know if your systems are running efficiently
- **Planning**: Understand usage patterns for capacity planning

Think of it as having eyes and ears on your infrastructure 24/7!

## 6A.2 Basic Logging - Your First Line of Defense

Logging is like keeping a diary of what your application is doing:

```python
# basic_logging.py
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('app.log')  # Save to file
    ]
)

logger = logging.getLogger(__name__)

def process_server_request(server_name):
    """Example function that logs its activities."""
    
    logger.info(f"Processing request for server: {server_name}")
    
    try:
        # Simulate some work
        if server_name == "broken-server":
            raise Exception("Server is not responding")
        
        # Simulate processing time
        import time
        time.sleep(0.1)
        
        logger.info(f"Successfully processed {server_name}")
        return {"status": "success", "server": server_name}
        
    except Exception as e:
        logger.error(f"Failed to process {server_name}: {e}")
        return {"status": "error", "server": server_name, "error": str(e)}

def main():
    """Demo different log levels."""
    
    logger.debug("This is a debug message (very detailed)")
    logger.info("This is an info message (general information)")
    logger.warning("This is a warning (something might be wrong)")
    logger.error("This is an error (something went wrong)")
    logger.critical("This is critical (system might be down!)")
    
    # Process some servers
    servers = ["web-01", "web-02", "broken-server", "db-01"]
    
    for server in servers:
        result = process_server_request(server)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
```

### Log Levels Explained:
- **DEBUG**: Very detailed information, usually only used when debugging
- **INFO**: General information about what's happening
- **WARNING**: Something unexpected happened, but the program is still working
- **ERROR**: Something went wrong, but the program can continue
- **CRITICAL**: Something very serious happened, the program might stop

### Exercise 6A.1: Your First Logger
1. Create `basic_logging.py` and run it
2. Check the `app.log` file that gets created
3. Change the logging level to `DEBUG` and see what happens
4. Add logging to a function that checks if a file exists

## 6A.3 System Resource Monitoring

Let's monitor the basic health of our system:

```python
# system_monitor.py
import psutil
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_system_info():
    """Get basic system information."""
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_total_gb = round(memory.total / (1024**3), 2)
    memory_used_gb = round(memory.used / (1024**3), 2)
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = round((disk.used / disk.total) * 100, 1)
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "usage_percent": cpu_percent
        },
        "memory": {
            "usage_percent": memory_percent,
            "total_gb": memory_total_gb,
            "used_gb": memory_used_gb
        },
        "disk": {
            "usage_percent": disk_percent,
            "total_gb": disk_total_gb,
            "free_gb": disk_free_gb
        }
    }

def check_system_health():
    """Check if system resources are within healthy limits."""
    
    info = get_system_info()
    alerts = []
    
    # Define thresholds (you can adjust these)
    CPU_WARNING = 80
    CPU_CRITICAL = 95
    MEMORY_WARNING = 85
    MEMORY_CRITICAL = 95
    DISK_WARNING = 80
    DISK_CRITICAL = 90
    
    # Check CPU
    cpu_usage = info["cpu"]["usage_percent"]
    if cpu_usage >= CPU_CRITICAL:
        alerts.append(f"CRITICAL: CPU usage is {cpu_usage}% (>= {CPU_CRITICAL}%)")
        logger.critical(f"CPU usage critical: {cpu_usage}%")
    elif cpu_usage >= CPU_WARNING:
        alerts.append(f"WARNING: CPU usage is {cpu_usage}% (>= {CPU_WARNING}%)")
        logger.warning(f"CPU usage high: {cpu_usage}%")
    
    # Check Memory
    memory_usage = info["memory"]["usage_percent"]
    if memory_usage >= MEMORY_CRITICAL:
        alerts.append(f"CRITICAL: Memory usage is {memory_usage}% (>= {MEMORY_CRITICAL}%)")
        logger.critical(f"Memory usage critical: {memory_usage}%")
    elif memory_usage >= MEMORY_WARNING:
        alerts.append(f"WARNING: Memory usage is {memory_usage}% (>= {MEMORY_WARNING}%)")
        logger.warning(f"Memory usage high: {memory_usage}%")
    
    # Check Disk
    disk_usage = info["disk"]["usage_percent"]
    if disk_usage >= DISK_CRITICAL:
        alerts.append(f"CRITICAL: Disk usage is {disk_usage}% (>= {DISK_CRITICAL}%)")
        logger.critical(f"Disk usage critical: {disk_usage}%")
    elif disk_usage >= DISK_WARNING:
        alerts.append(f"WARNING: Disk usage is {disk_usage}% (>= {DISK_WARNING}%)")
        logger.warning(f"Disk usage high: {disk_usage}%")
    
    return info, alerts

def print_system_report():
    """Print a formatted system report."""
    
    info, alerts = check_system_health()
    
    print("=" * 50)
    print("SYSTEM HEALTH REPORT")
    print("=" * 50)
    print(f"Time: {info['timestamp']}")
    print()
    
    print(f"CPU Usage: {info['cpu']['usage_percent']:.1f}%")
    print(f"Memory Usage: {info['memory']['usage_percent']:.1f}% ({info['memory']['used_gb']:.1f}GB / {info['memory']['total_gb']:.1f}GB)")
    print(f"Disk Usage: {info['disk']['usage_percent']:.1f}% ({info['disk']['free_gb']:.1f}GB free)")
    
    if alerts:
        print("\n🚨 ALERTS:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\n✅ All systems normal")
    
    return info, alerts

def monitor_continuously(duration_minutes=5, check_interval=10):
    """Monitor system continuously for a specified duration."""
    
    print(f"Starting continuous monitoring for {duration_minutes} minutes...")
    print(f"Checking every {check_interval} seconds")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            print_system_report()
            
            # Wait before next check
            print(f"\nWaiting {check_interval} seconds for next check...")
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("\nMonitoring completed!")

if __name__ == "__main__":
    # Run a single system check
    print_system_report()
    
    # Uncomment to run continuous monitoring:
    # monitor_continuously(duration_minutes=2, check_interval=5)
```

### Exercise 6A.2: System Monitor
1. Install psutil: `pip install psutil`
2. Create and run `system_monitor.py`
3. Try running a CPU-intensive task (like a large calculation) and see how the monitor reacts
4. Modify the thresholds to make them more or less sensitive
5. Add monitoring for network usage

## 6A.4 Simple Error Handling and Retry Logic

When things go wrong, handle it gracefully:

```python
# error_handling.py
import time
import random
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def unreliable_service_call(service_name: str) -> dict:
    """Simulate an unreliable service that sometimes fails."""
    
    # Simulate random failures
    if random.random() < 0.3:  # 30% chance of failure
        raise ConnectionError(f"Could not connect to {service_name}")
    
    if random.random() < 0.2:  # 20% chance of timeout
        time.sleep(3)  # Simulate slow response
        raise TimeoutError(f"Timeout connecting to {service_name}")
    
    # Success case
    return {"status": "success", "service": service_name, "data": "some data"}

def safe_service_call(service_name: str, max_retries: int = 3, timeout: int = 2) -> Optional[dict]:
    """Make a service call with retry logic and error handling."""
    
    logger.info(f"Calling service: {service_name}")
    
    for attempt in range(max_retries + 1):
        try:
            # Log the attempt
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for {service_name}")
            
            # Make the service call
            result = unreliable_service_call(service_name)
            
            logger.info(f"Successfully called {service_name} on attempt {attempt + 1}")
            return result
        
        except ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries reached for {service_name}")
                return None
        
        except TimeoutError as e:
            logger.warning(f"Timeout error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries:
                wait_time = 1
                logger.info(f"Waiting {wait_time} second before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries reached for {service_name}")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error calling {service_name}: {e}")
            return None
    
    return None

def monitor_multiple_services():
    """Monitor multiple services and report their status."""
    
    services = ["user-service", "payment-service", "inventory-service", "email-service"]
    
    print("Service Health Check Report")
    print("=" * 40)
    
    healthy_services = 0
    total_services = len(services)
    
    for service in services:
        print(f"\nChecking {service}...")
        
        result = safe_service_call(service, max_retries=2)
        
        if result:
            print(f"✅ {service}: HEALTHY")
            healthy_services += 1
        else:
            print(f"❌ {service}: UNHEALTHY")
    
    print(f"\nSummary: {healthy_services}/{total_services} services are healthy")
    
    if healthy_services == total_services:
        print("🎉 All services are running perfectly!")
    elif healthy_services == 0:
        print("🚨 CRITICAL: All services are down!")
    else:
        print("⚠️  Some services need attention")

if __name__ == "__main__":
    # Test a single service call
    print("Testing single service call:")
    result = safe_service_call("test-service")
    print(f"Result: {result}")
    print()
    
    # Monitor multiple services
    monitor_multiple_services()
```

### Exercise 6A.3: Error Handling
1. Create and run `error_handling.py` multiple times
2. Notice how the retry logic works with different types of errors
3. Modify the failure rates and see how it affects the results
4. Add a new error type (like "Service Unavailable") and handle it

## 6A.5 Simple Alerting System

Create a basic system to send alerts when problems occur:

```python
# simple_alerts.py
import smtplib
import json
import requests
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AlertManager:
    """Simple alert management system."""
    
    def __init__(self):
        self.alerts = []
    
    def create_alert(self, title: str, message: str, severity: str = "warning"):
        """Create a new alert."""
        
        alert = {
            "id": len(self.alerts) + 1,
            "title": title,
            "message": message,
            "severity": severity,
            "created_at": datetime.now().isoformat(),
            "status": "open"
        }
        
        self.alerts.append(alert)
        logger.info(f"Alert created: {title} ({severity})")
        
        # Send notification based on severity
        if severity in ["critical", "error"]:
            self.send_notification(alert)
        
        return alert
    
    def send_notification(self, alert: Dict):
        """Send alert notification (placeholder for real implementation)."""
        
        print(f"\n🚨 ALERT NOTIFICATION 🚨")
        print(f"Title: {alert['title']}")
        print(f"Severity: {alert['severity'].upper()}")
        print(f"Message: {alert['message']}")
        print(f"Time: {alert['created_at']}")
        print("-" * 40)
        
        # In a real system, you would send this to:
        # - Slack/Discord webhook
        # - Email
        # - SMS service
        # - PagerDuty
        
        logger.info(f"Notification sent for alert: {alert['id']}")
    
    def send_to_webhook(self, alert: Dict, webhook_url: str):
        """Send alert to a webhook (like Slack)."""
        
        try:
            payload = {
                "text": f"🚨 Alert: {alert['title']}",
                "attachments": [
                    {
                        "color": "danger" if alert['severity'] == "critical" else "warning",
                        "fields": [
                            {"title": "Severity", "value": alert['severity'], "short": True},
                            {"title": "Message", "value": alert['message'], "short": False},
                            {"title": "Time", "value": alert['created_at'], "short": True}
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Alert sent to webhook successfully")
            else:
                logger.error(f"Failed to send webhook: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all open alerts."""
        return [alert for alert in self.alerts if alert["status"] == "open"]
    
    def resolve_alert(self, alert_id: int):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["status"] = "resolved"
                alert["resolved_at"] = datetime.now().isoformat()
                logger.info(f"Alert {alert_id} resolved")
                return True
        
        logger.warning(f"Alert {alert_id} not found")
        return False

def monitor_with_alerts():
    """Example monitoring function that creates alerts."""
    
    alert_manager = AlertManager()
    
    # Simulate various system conditions
    scenarios = [
        ("CPU usage normal", "CPU usage is 25%", "info"),
        ("High memory usage", "Memory usage is 87% - consider adding more RAM", "warning"), 
        ("Disk space critical", "Disk usage is 95% - immediate action required!", "critical"),
        ("Service down", "Payment service is not responding", "error"),
        ("Database slow", "Database queries taking >5 seconds", "warning"),
    ]
    
    print("Running monitoring scenarios...")
    print("=" * 50)
    
    for title, message, severity in scenarios:
        alert = alert_manager.create_alert(title, message, severity)
        
        # Simulate some time passing
        import time
        time.sleep(1)
    
    # Show all alerts
    print(f"\nActive Alerts ({len(alert_manager.get_active_alerts())}):")
    for alert in alert_manager.get_active_alerts():
        severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
        print(f"  {severity_icon} [{alert['id']}] {alert['title']} ({alert['severity']})")
    
    # Resolve some alerts
    print("\nResolving disk space alert...")
    alert_manager.resolve_alert(3)
    
    print(f"\nRemaining Active Alerts ({len(alert_manager.get_active_alerts())}):")
    for alert in alert_manager.get_active_alerts():
        severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
        print(f"  {severity_icon} [{alert['id']}] {alert['title']} ({alert['severity']})")

def test_webhook_alert():
    """Test sending an alert to a webhook."""
    
    alert_manager = AlertManager()
    
    # Create a test alert
    alert = alert_manager.create_alert(
        "Test Alert",
        "This is a test alert to verify webhook functionality",
        "warning"
    )
    
    # Send to httpbin (for testing)
    webhook_url = "https://httpbin.org/post"
    alert_manager.send_to_webhook(alert, webhook_url)
    
    print("Test webhook sent! Check the logs for results.")

if __name__ == "__main__":
    # Run monitoring with alerts
    monitor_with_alerts()
    
    print("\n" + "=" * 50)
    
    # Test webhook (uncomment to try)
    # test_webhook_alert()
```

### Exercise 6A.4: Alerting System
1. Create and run `simple_alerts.py`
2. Try creating alerts with different severities
3. Modify the code to send alerts to a real webhook URL (you can use https://httpbin.org/post for testing)
4. Add a function to save alerts to a JSON file
5. Create a simple dashboard that shows all active alerts

## 6A.6 Mini Project: Simple Health Dashboard

Combine everything into a basic health monitoring system:

```python
# health_dashboard.py
import psutil
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HealthDashboard:
    """Simple health monitoring dashboard."""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.services = {
            "web-server": {"status": "running", "last_check": datetime.now()},
            "database": {"status": "running", "last_check": datetime.now()},
            "cache": {"status": "running", "last_check": datetime.now()},
        }
    
    def collect_system_metrics(self) -> Dict:
        """Collect current system metrics."""
        
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": round((psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100, 1),
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
            }
            
            # Store in history (keep last 100 readings)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            logger.debug(f"Collected metrics: CPU={metrics['cpu_percent']}%, Memory={metrics['memory_percent']}%")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}
    
    def check_thresholds(self, metrics: Dict):
        """Check if any metrics exceed thresholds."""
        
        if not metrics:
            return
        
        # Define thresholds
        thresholds = {
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 85, "critical": 95},
            "disk_percent": {"warning": 80, "critical": 90}
        }
        
        for metric_name, value in metrics.items():
            if metric_name in thresholds and isinstance(value, (int, float)):
                
                threshold = thresholds[metric_name]
                
                if value >= threshold["critical"]:
                    self.create_alert(
                        f"Critical {metric_name.replace('_', ' ').title()}",
                        f"{metric_name.replace('_', ' ').title()} is {value}% (>= {threshold['critical']}%)",
                        "critical"
                    )
                elif value >= threshold["warning"]:
                    self.create_alert(
                        f"High {metric_name.replace('_', ' ').title()}",
                        f"{metric_name.replace('_', ' ').title()} is {value}% (>= {threshold['warning']}%)",
                        "warning"
                    )
    
    def create_alert(self, title: str, message: str, severity: str):
        """Create a new alert."""
        
        # Check if similar alert already exists
        for alert in self.alerts:
            if alert["title"] == title and alert["status"] == "open":
                logger.debug(f"Similar alert already exists: {title}")
                return
        
        alert = {
            "id": len(self.alerts) + 1,
            "title": title,
            "message": message,
            "severity": severity,
            "status": "open",
            "created_at": datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {title} ({severity})")
    
    def get_system_summary(self) -> Dict:
        """Get a summary of system health."""
        
        if not self.metrics_history:
            return {"status": "unknown", "message": "No metrics available"}
        
        latest_metrics = self.metrics_history[-1]
        active_alerts = [a for a in self.alerts if a["status"] == "open"]
        
        # Determine overall status
        critical_alerts = [a for a in active_alerts if a["severity"] == "critical"]
        warning_alerts = [a for a in active_alerts if a["severity"] == "warning"]
        
        if critical_alerts:
            status = "critical"
            message = f"{len(critical_alerts)} critical alerts"
        elif warning_alerts:
            status = "warning"
            message = f"{len(warning_alerts)} warnings"
        else:
            status = "healthy"
            message = "All systems normal"
        
        return {
            "status": status,
            "message": message,
            "metrics": latest_metrics,
            "alerts": {
                "total": len(active_alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts)
            },
            "services": len(self.services)
        }
    
    def print_dashboard(self):
        """Print a formatted dashboard."""
        
        summary = self.get_system_summary()
        
        # Header
        print("\n" + "=" * 60)
        print("SYSTEM HEALTH DASHBOARD")
        print("=" * 60)
        print(f"Status: {summary['status'].upper()} - {summary['message']}")
        print(f"Last Updated: {summary['metrics'].get('timestamp', 'Unknown')}")
        print()
        
        # System Metrics
        if summary.get('metrics'):
            metrics = summary['metrics']
            print("SYSTEM METRICS:")
            print(f"  CPU Usage:    {metrics.get('cpu_percent', 'N/A')}%")
            print(f"  Memory Usage: {metrics.get('memory_percent', 'N/A')}%")
            print(f"  Disk Usage:   {metrics.get('disk_percent', 'N/A')}%")
            if metrics.get('load_average'):
                print(f"  Load Average: {metrics['load_average']:.2f}")
            print()
        
        # Active Alerts
        active_alerts = [a for a in self.alerts if a["status"] == "open"]
        
        if active_alerts:
            print("ACTIVE ALERTS:")
            for alert in active_alerts[-5:]:  # Show last 5 alerts
                severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
                print(f"  {severity_icon} [{alert['severity'].upper()}] {alert['title']}")
                print(f"     {alert['message']}")
            
            if len(active_alerts) > 5:
                print(f"     ... and {len(active_alerts) - 5} more alerts")
        else:
            print("ACTIVE ALERTS: None ✅")
        
        print()
        
        # Services Status
        print("SERVICES:")
        for service_name, service_info in self.services.items():
            status_icon = "✅" if service_info["status"] == "running" else "❌"
            print(f"  {status_icon} {service_name}: {service_info['status']}")
        
        print("=" * 60)
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        
        # Collect metrics
        metrics = self.collect_system_metrics()
        
        # Check thresholds
        self.check_thresholds(metrics)
        
        # Print dashboard
        self.print_dashboard()
    
    def run_continuous_monitoring(self, duration_minutes: int = 5, interval_seconds: int = 10):
        """Run continuous monitoring."""
        
        print(f"Starting continuous monitoring for {duration_minutes} minutes...")
        print(f"Checking every {interval_seconds} seconds")
        print("Press Ctrl+C to stop")
        
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                self.run_monitoring_cycle()
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        print("\nMonitoring completed!")

def main():
    """Run the health dashboard."""
    
    dashboard = HealthDashboard()
    
    print("Health Dashboard Demo")
    print("Options:")
    print("1. Single check")
    print("2. Continuous monitoring (5 minutes)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        dashboard.run_monitoring_cycle()
    elif choice == "2":
        dashboard.run_continuous_monitoring(duration_minutes=2, interval_seconds=5)
    else:
        print("Running single check...")
        dashboard.run_monitoring_cycle()

if __name__ == "__main__":
    main()
```

### Exercise 6A.5: Health Dashboard
1. Create and run `health_dashboard.py`
2. Try both single check and continuous monitoring
3. Load your system (open many programs) and watch the alerts
4. Add monitoring for network connections or specific processes
5. Save the metrics history to a JSON file

## Key Takeaways

- **Logging is essential**: Always log what your application is doing
- **Monitor key metrics**: CPU, memory, disk, and network are the basics
- **Set appropriate thresholds**: Not too sensitive, not too relaxed
- **Handle errors gracefully**: Retry logic and fallbacks are crucial
- **Alert on what matters**: Too many alerts = ignored alerts

## Next Steps

In Chapter 6B, you'll learn about:
- Advanced monitoring with external services
- Database integration for metrics storage
- Creating web dashboards
- Integration with monitoring platforms like Prometheus

Remember: Start simple and build up. A basic monitoring system that works is better than a complex one that doesn't!