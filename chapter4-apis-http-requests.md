# Chapter 4: Working with APIs and HTTP Requests

## Learning Objectives
- Understand HTTP fundamentals and RESTful API concepts
- Master the requests library for API interactions
- Implement authentication methods (API keys, OAuth, JWT)
- Handle errors, retries, and rate limiting
- Build monitoring and health check systems
- Create tools for service discovery and integration

## 4.1 HTTP Fundamentals for DevOps

### Understanding HTTP Methods and Status Codes

```python
import requests
from typing import Dict, Any, Optional
from enum import Enum

class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT" 
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class HTTPStatus:
    """Common HTTP status codes for API monitoring."""
    
    # Success
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client Errors
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    # Server Errors
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504

def analyze_http_response(response: requests.Response) -> Dict[str, Any]:
    """Analyze HTTP response for monitoring purposes."""
    return {
        'status_code': response.status_code,
        'status_text': response.reason,
        'headers': dict(response.headers),
        'response_time_ms': response.elapsed.total_seconds() * 1000,
        'content_length': len(response.content),
        'content_type': response.headers.get('content-type', ''),
        'is_success': response.status_code < 400,
        'is_redirect': 300 <= response.status_code < 400,
        'is_client_error': 400 <= response.status_code < 500,
        'is_server_error': response.status_code >= 500
    }
```

### Basic API Interactions

```python
import requests
import json
from datetime import datetime
from typing import Optional, Union

class APIClient:
    """Generic API client for service interactions."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'DevOps-Monitor/1.0',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            return response
        except requests.exceptions.Timeout:
            raise APIError(f"Request timeout for {url}")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Connection error for {url}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed for {url}: {str(e)}")
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """Make GET request."""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> requests.Response:
        """Make POST request."""
        return self._make_request('POST', endpoint, data=data, json=json_data)
    
    def put(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> requests.Response:
        """Make PUT request."""
        return self._make_request('PUT', endpoint, data=data, json=json_data)
    
    def delete(self, endpoint: str) -> requests.Response:
        """Make DELETE request."""
        return self._make_request('DELETE', endpoint)

class APIError(Exception):
    """Custom exception for API errors."""
    pass

# Example usage
def test_api_endpoints():
    """Test basic API functionality."""
    client = APIClient('https://jsonplaceholder.typicode.com')
    
    try:
        # GET request
        response = client.get('/posts/1')
        if response.status_code == 200:
            post_data = response.json()
            print(f"Retrieved post: {post_data['title']}")
        
        # POST request
        new_post = {
            'title': 'Test Post from DevOps',
            'body': 'This is a test post for monitoring',
            'userId': 1
        }
        
        response = client.post('/posts', json_data=new_post)
        if response.status_code == 201:
            created_post = response.json()
            print(f"Created post with ID: {created_post['id']}")
    
    except APIError as e:
        print(f"API Error: {e}")
```

## 4.2 Authentication Methods

### API Key Authentication

```python
class APIKeyAuth:
    """Handle API key authentication in headers or query parameters."""
    
    def __init__(self, api_key: str, header_name: str = 'X-API-Key'):
        self.api_key = api_key
        self.header_name = header_name
    
    def apply_to_session(self, session: requests.Session):
        """Apply API key to session headers."""
        session.headers[self.header_name] = self.api_key
    
    def apply_to_params(self, params: Dict[str, str]) -> Dict[str, str]:
        """Apply API key to query parameters."""
        params = params or {}
        params['api_key'] = self.api_key
        return params

# Example with cloud provider APIs
class AWSAPIClient(APIClient):
    """AWS API client with proper authentication."""
    
    def __init__(self, base_url: str, access_key: str, secret_key: str):
        super().__init__(base_url)
        self.access_key = access_key
        self.secret_key = secret_key
        
        # AWS requires special signature authentication
        # This is simplified - use boto3 for real AWS interactions
        self.session.headers.update({
            'X-AWS-Access-Key': access_key
        })

class DigitalOceanClient(APIClient):
    """DigitalOcean API client."""
    
    def __init__(self, api_token: str):
        super().__init__('https://api.digitalocean.com/v2')
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}'
        })
    
    def list_droplets(self) -> List[Dict[str, Any]]:
        """List all droplets."""
        response = self.get('/droplets')
        if response.status_code == 200:
            return response.json()['droplets']
        else:
            raise APIError(f"Failed to list droplets: {response.status_code}")
    
    def create_droplet(self, name: str, region: str, size: str, image: str) -> Dict[str, Any]:
        """Create a new droplet."""
        droplet_config = {
            'name': name,
            'region': region,
            'size': size,
            'image': image,
            'ssh_keys': [],
            'backups': False,
            'ipv6': True,
            'user_data': None,
            'monitoring': True
        }
        
        response = self.post('/droplets', json_data=droplet_config)
        if response.status_code == 202:
            return response.json()['droplet']
        else:
            raise APIError(f"Failed to create droplet: {response.status_code}")
```

### JWT Token Authentication

```python
import jwt
from datetime import datetime, timedelta
import base64
import hashlib
import hmac

class JWTAuth:
    """Handle JWT token authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token = None
        self.expires_at = None
    
    def generate_token(self, payload: Dict[str, Any], expires_in_hours: int = 24) -> str:
        """Generate JWT token with expiration."""
        payload['exp'] = datetime.utcnow() + timedelta(hours=expires_in_hours)
        payload['iat'] = datetime.utcnow()
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self.token = token
        self.expires_at = payload['exp']
        
        return token
    
    def is_token_valid(self) -> bool:
        """Check if current token is still valid."""
        if not self.token or not self.expires_at:
            return False
        
        return datetime.utcnow() < self.expires_at
    
    def apply_to_session(self, session: requests.Session):
        """Apply JWT token to session headers."""
        if self.is_token_valid():
            session.headers['Authorization'] = f'Bearer {self.token}'
        else:
            raise APIError("JWT token is expired or invalid")

class ServiceAuthClient(APIClient):
    """Client for services using JWT authentication."""
    
    def __init__(self, base_url: str, username: str, password: str):
        super().__init__(base_url)
        self.username = username
        self.password = password
        self.jwt_auth = None
    
    def authenticate(self) -> bool:
        """Authenticate with service and get JWT token."""
        auth_data = {
            'username': self.username,
            'password': self.password
        }
        
        response = self.post('/auth/login', json_data=auth_data)
        if response.status_code == 200:
            auth_response = response.json()
            token = auth_response.get('access_token')
            
            if token:
                self.session.headers['Authorization'] = f'Bearer {token}'
                return True
        
        return False
    
    def ensure_authenticated(self):
        """Ensure client is authenticated before making requests."""
        if 'Authorization' not in self.session.headers:
            if not self.authenticate():
                raise APIError("Failed to authenticate with service")
```

## 4.3 Error Handling and Retries

### Robust Error Handling

```python
import time
from functools import wraps
from typing import Callable, List, Type
import logging

logger = logging.getLogger(__name__)

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 backoff_factor: float = 2.0,
                 retry_status_codes: List[int] = None):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_status_codes = retry_status_codes or [429, 502, 503, 504]

def retry_on_failure(retry_config: RetryConfig):
    """Decorator for retrying failed API calls."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    response = func(*args, **kwargs)
                    
                    # Check if we should retry based on status code
                    if hasattr(response, 'status_code') and response.status_code in retry_config.retry_status_codes:
                        if attempt < retry_config.max_retries:
                            sleep_time = retry_config.backoff_factor ** attempt
                            logger.warning(f"HTTP {response.status_code} received, retrying in {sleep_time}s (attempt {attempt + 1})")
                            time.sleep(sleep_time)
                            continue
                    
                    return response
                
                except (requests.exceptions.RequestException, APIError) as e:
                    last_exception = e
                    
                    if attempt < retry_config.max_retries:
                        sleep_time = retry_config.backoff_factor ** attempt
                        logger.warning(f"Request failed: {e}, retrying in {sleep_time}s (attempt {attempt + 1})")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Request failed after {retry_config.max_retries} retries: {e}")
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator

class RobustAPIClient(APIClient):
    """API client with built-in retry logic and error handling."""
    
    def __init__(self, base_url: str, retry_config: RetryConfig = None):
        super().__init__(base_url)
        self.retry_config = retry_config or RetryConfig()
    
    @retry_on_failure(RetryConfig())
    def get_with_retry(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """GET request with automatic retries."""
        return self.get(endpoint, params)
    
    @retry_on_failure(RetryConfig())
    def post_with_retry(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> requests.Response:
        """POST request with automatic retries."""
        return self.post(endpoint, data, json_data)
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health with comprehensive error handling."""
        start_time = time.time()
        
        try:
            response = self.get_with_retry('/health')
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat(),
                'details': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
            }
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat()
            }
```

### Rate Limiting

```python
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str = 'default') -> bool:
        """Check if request is allowed under rate limit."""
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.time_window)
            
            # Remove old requests
            while self.requests[identifier] and self.requests[identifier][0] < cutoff:
                self.requests[identifier].popleft()
            
            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            
            return False
    
    def wait_time(self, identifier: str = 'default') -> float:
        """Get wait time until next request is allowed."""
        with self.lock:
            if len(self.requests[identifier]) < self.max_requests:
                return 0.0
            
            oldest_request = self.requests[identifier][0]
            time_to_wait = self.time_window - (datetime.now() - oldest_request).total_seconds()
            return max(0.0, time_to_wait)

class RateLimitedAPIClient(RobustAPIClient):
    """API client with rate limiting."""
    
    def __init__(self, base_url: str, max_requests: int = 100, time_window: int = 60):
        super().__init__(base_url)
        self.rate_limiter = RateLimiter(max_requests, time_window)
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded."""
        if not self.rate_limiter.is_allowed():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.info(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> requests.Response:
        """GET request with rate limiting."""
        self._wait_for_rate_limit()
        return super().get(endpoint, params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None) -> requests.Response:
        """POST request with rate limiting."""
        self._wait_for_rate_limit()
        return super().post(endpoint, data, json_data)
```

## 4.4 Service Monitoring and Health Checks

### Health Check System

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import yaml

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ServiceMonitor:
    """Monitor multiple services and their health."""
    
    def __init__(self, config_file: str = "config/services.yaml"):
        self.services = self._load_service_config(config_file)
        self.results = []
    
    def _load_service_config(self, config_file: str) -> List[Dict[str, Any]]:
        """Load service configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('services', [])
        except FileNotFoundError:
            # Create default config
            default_config = {
                'services': [
                    {
                        'name': 'web-api',
                        'url': 'https://api.example.com/health',
                        'method': 'GET',
                        'timeout': 10,
                        'expected_status': 200,
                        'headers': {'User-Agent': 'ServiceMonitor/1.0'}
                    },
                    {
                        'name': 'database',
                        'url': 'http://db.example.com:5432/health',
                        'method': 'GET',
                        'timeout': 5,
                        'expected_status': 200
                    }
                ]
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, indent=2)
            
            return default_config['services']
    
    def check_service_health(self, service_config: Dict[str, Any]) -> HealthCheckResult:
        """Check health of a single service."""
        name = service_config['name']
        url = service_config['url']
        method = service_config.get('method', 'GET')
        timeout = service_config.get('timeout', 10)
        expected_status = service_config.get('expected_status', 200)
        headers = service_config.get('headers', {})
        
        start_time = time.time()
        
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                headers=headers
            )
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on response
            if response.status_code == expected_status:
                status = ServiceStatus.HEALTHY
            elif 200 <= response.status_code < 400:
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.UNHEALTHY
            
            # Try to parse JSON response for additional details
            details = None
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    details = response.json()
            except:
                pass
            
            return HealthCheckResult(
                service_name=name,
                status=status,
                response_time_ms=response_time,
                status_code=response.status_code,
                details=details
            )
        
        except requests.exceptions.Timeout:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message=f"Request timeout after {timeout}s"
            )
        
        except requests.exceptions.ConnectionError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message="Connection failed"
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=name,
                status=ServiceStatus.UNKNOWN,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def check_all_services(self, parallel: bool = True) -> List[HealthCheckResult]:
        """Check health of all configured services."""
        if parallel:
            return self._check_services_parallel()
        else:
            return self._check_services_sequential()
    
    def _check_services_parallel(self) -> List[HealthCheckResult]:
        """Check services in parallel using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_service = {
                executor.submit(self.check_service_health, service): service
                for service in self.services
            }
            
            for future in as_completed(future_to_service):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    service = future_to_service[future]
                    results.append(HealthCheckResult(
                        service_name=service['name'],
                        status=ServiceStatus.UNKNOWN,
                        response_time_ms=0,
                        error_message=f"Health check failed: {str(e)}"
                    ))
        
        self.results = results
        return results
    
    def _check_services_sequential(self) -> List[HealthCheckResult]:
        """Check services sequentially."""
        results = []
        
        for service in self.services:
            result = self.check_service_health(service)
            results.append(result)
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if not self.results:
            return {'error': 'No health check results available'}
        
        healthy_count = sum(1 for r in self.results if r.status == ServiceStatus.HEALTHY)
        unhealthy_count = sum(1 for r in self.results if r.status == ServiceStatus.UNHEALTHY)
        degraded_count = sum(1 for r in self.results if r.status == ServiceStatus.DEGRADED)
        
        avg_response_time = sum(r.response_time_ms for r in self.results) / len(self.results)
        max_response_time = max(r.response_time_ms for r in self.results)
        
        return {
            'summary': {
                'total_services': len(self.results),
                'healthy': healthy_count,
                'unhealthy': unhealthy_count,
                'degraded': degraded_count,
                'overall_health': 'healthy' if unhealthy_count == 0 else 'degraded' if degraded_count > 0 else 'unhealthy'
            },
            'performance': {
                'average_response_time_ms': avg_response_time,
                'max_response_time_ms': max_response_time
            },
            'services': [
                {
                    'name': r.service_name,
                    'status': r.status.value,
                    'response_time_ms': r.response_time_ms,
                    'status_code': r.status_code,
                    'error_message': r.error_message,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_unhealthy_services(self) -> List[HealthCheckResult]:
        """Get list of unhealthy services."""
        return [r for r in self.results if r.status == ServiceStatus.UNHEALTHY]
```

## Exercise 4: Create a Service Health Monitoring System

### Exercise Overview
Build a comprehensive service monitoring system that checks multiple APIs, handles authentication, implements retries, and generates alerts.

### Step 1: Project Setup

```bash
mkdir service-monitor
cd service-monitor
uv init
uv add requests pyyaml click tabulate
mkdir -p {config,alerts,reports}
```

### Step 2: Create Configuration

Create `config/services.yaml`:

```yaml
services:
  - name: "github-api"
    url: "https://api.github.com"
    method: "GET"
    timeout: 10
    expected_status: 200
    headers:
      User-Agent: "ServiceMonitor/1.0"
  
  - name: "httpbin-status"
    url: "https://httpbin.org/status/200"
    method: "GET"
    timeout: 5
    expected_status: 200
  
  - name: "httpbin-delay"
    url: "https://httpbin.org/delay/2"
    method: "GET"
    timeout: 10
    expected_status: 200
  
  - name: "intentional-failure"
    url: "https://httpbin.org/status/500"
    method: "GET"
    timeout: 5
    expected_status: 200

monitoring:
  check_interval: 60  # seconds
  parallel_checks: true
  max_workers: 10

alerts:
  webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  email_enabled: false
  thresholds:
    response_time_warning: 1000  # ms
    response_time_critical: 5000  # ms
    failure_count_alert: 3
```

### Step 3: Create Alert System

Create `src/service_monitor/alerts.py`:

```python
import requests
import json
from typing import List, Dict, Any
from datetime import datetime
from .monitor import HealthCheckResult, ServiceStatus

class AlertManager:
    """Manage alerts for service monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = []
    
    def should_alert(self, result: HealthCheckResult) -> bool:
        """Determine if an alert should be sent."""
        alert_config = self.config.get('alerts', {})
        thresholds = alert_config.get('thresholds', {})
        
        # Alert on unhealthy status
        if result.status == ServiceStatus.UNHEALTHY:
            return True
        
        # Alert on slow response times
        warning_threshold = thresholds.get('response_time_warning', 1000)
        critical_threshold = thresholds.get('response_time_critical', 5000)
        
        if result.response_time_ms > critical_threshold:
            return True
        
        return False
    
    def generate_alert_message(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate alert message from health check results."""
        unhealthy_services = [r for r in results if r.status == ServiceStatus.UNHEALTHY]
        slow_services = [r for r in results if r.response_time_ms > 1000]
        
        if not unhealthy_services and not slow_services:
            return None
        
        message = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'service_health',
            'summary': f"Found {len(unhealthy_services)} unhealthy and {len(slow_services)} slow services",
            'details': {
                'unhealthy_services': [
                    {
                        'name': r.service_name,
                        'error': r.error_message,
                        'status_code': r.status_code
                    }
                    for r in unhealthy_services
                ],
                'slow_services': [
                    {
                        'name': r.service_name,
                        'response_time_ms': r.response_time_ms
                    }
                    for r in slow_services
                ]
            }
        }
        
        return message
    
    def send_slack_alert(self, message: Dict[str, Any]) -> bool:
        """Send alert to Slack webhook."""
        webhook_url = self.config.get('alerts', {}).get('webhook_url')
        if not webhook_url:
            return False
        
        slack_message = {
            'text': f"🚨 Service Health Alert: {message['summary']}",
            'attachments': [
                {
                    'color': 'danger',
                    'fields': [
                        {
                            'title': 'Unhealthy Services',
                            'value': '\n'.join([
                                f"• {service['name']}: {service['error']}"
                                for service in message['details']['unhealthy_services']
                            ]) or 'None',
                            'short': True
                        },
                        {
                            'title': 'Slow Services',
                            'value': '\n'.join([
                                f"• {service['name']}: {service['response_time_ms']:.0f}ms"
                                for service in message['details']['slow_services']
                            ]) or 'None',
                            'short': True
                        }
                    ],
                    'footer': 'Service Monitor',
                    'ts': int(datetime.now().timestamp())
                }
            ]
        }
        
        try:
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False
    
    def process_results(self, results: List[HealthCheckResult]):
        """Process health check results and send alerts if needed."""
        alert_message = self.generate_alert_message(results)
        
        if alert_message:
            # Send to Slack
            if self.send_slack_alert(alert_message):
                print("Alert sent to Slack")
            
            # Store in history
            self.alert_history.append(alert_message)
            
            # Optionally save to file
            self.save_alert_to_file(alert_message)
    
    def save_alert_to_file(self, alert: Dict[str, Any]):
        """Save alert to file for audit trail."""
        filename = f"alerts/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(alert, f, indent=2)
        except Exception as e:
            print(f"Failed to save alert to file: {e}")
```

### Step 4: Create CLI Interface

Create `src/service_monitor/cli.py`:

```python
#!/usr/bin/env python3
import click
import time
import yaml
from pathlib import Path
from tabulate import tabulate

from .monitor import ServiceMonitor, ServiceStatus
from .alerts import AlertManager

@click.group()
def cli():
    """Service Health Monitoring System."""
    pass

@cli.command()
@click.option('--config', '-c', default='config/services.yaml', help='Configuration file')
@click.option('--parallel/--sequential', default=True, help='Run checks in parallel')
@click.option('--output', '-o', help='Save report to file')
def check(config, parallel, output):
    """Run health checks on all configured services."""
    click.echo("Starting service health checks...")
    
    monitor = ServiceMonitor(config)
    results = monitor.check_all_services(parallel=parallel)
    
    # Display results in table format
    table_data = []
    for result in results:
        status_icon = {
            ServiceStatus.HEALTHY: "✅",
            ServiceStatus.UNHEALTHY: "❌", 
            ServiceStatus.DEGRADED: "⚠️",
            ServiceStatus.UNKNOWN: "❓"
        }.get(result.status, "❓")
        
        table_data.append([
            result.service_name,
            f"{status_icon} {result.status.value}",
            f"{result.response_time_ms:.0f}ms",
            result.status_code or "N/A",
            result.error_message or ""
        ])
    
    headers = ["Service", "Status", "Response Time", "HTTP Code", "Error"]
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Generate and display summary
    report = monitor.generate_report()
    summary = report['summary']
    
    click.echo(f"\nSummary:")
    click.echo(f"  Total Services: {summary['total_services']}")
    click.echo(f"  Healthy: {summary['healthy']}")
    click.echo(f"  Unhealthy: {summary['unhealthy']}")
    click.echo(f"  Degraded: {summary['degraded']}")
    click.echo(f"  Overall Health: {summary['overall_health']}")
    
    # Save report if requested
    if output:
        with open(output, 'w') as f:
            yaml.dump(report, f, indent=2)
        click.echo(f"Report saved to: {output}")
    
    # Process alerts
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    alert_manager = AlertManager(config_data)
    alert_manager.process_results(results)

@cli.command()
@click.option('--config', '-c', default='config/services.yaml', help='Configuration file')
@click.option('--interval', '-i', type=int, help='Check interval in seconds')
def monitor(config, interval):
    """Continuously monitor services."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    check_interval = interval or config_data.get('monitoring', {}).get('check_interval', 60)
    
    click.echo(f"Starting continuous monitoring (interval: {check_interval}s)")
    click.echo("Press Ctrl+C to stop")
    
    monitor = ServiceMonitor(config)
    alert_manager = AlertManager(config_data)
    
    try:
        while True:
            click.echo(f"\n{'-'*50}")
            click.echo(f"Running health checks at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            results = monitor.check_all_services(parallel=True)
            
            # Quick summary
            healthy = sum(1 for r in results if r.status == ServiceStatus.HEALTHY)
            unhealthy = sum(1 for r in results if r.status == ServiceStatus.UNHEALTHY)
            
            click.echo(f"Results: {healthy} healthy, {unhealthy} unhealthy")
            
            # Show unhealthy services
            unhealthy_services = [r for r in results if r.status == ServiceStatus.UNHEALTHY]
            if unhealthy_services:
                click.echo("Unhealthy services:")
                for result in unhealthy_services:
                    click.echo(f"  ❌ {result.service_name}: {result.error_message}")
            
            # Process alerts
            alert_manager.process_results(results)
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped")

@cli.command()
@click.argument('service_name')
@click.argument('service_url')
@click.option('--config', '-c', default='config/services.yaml', help='Configuration file')
@click.option('--method', default='GET', help='HTTP method')
@click.option('--timeout', default=10, help='Request timeout')
def add_service(service_name, service_url, config, method, timeout):
    """Add a new service to monitor."""
    # Load existing config
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {'services': []}
    
    # Add new service
    new_service = {
        'name': service_name,
        'url': service_url,
        'method': method,
        'timeout': timeout,
        'expected_status': 200
    }
    
    config_data['services'].append(new_service)
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, indent=2)
    
    click.echo(f"Added service '{service_name}' to monitoring configuration")

@cli.command()
@click.argument('url')
@click.option('--timeout', default=10, help='Request timeout')
def test_endpoint(url, timeout):
    """Test a single endpoint quickly."""
    from .monitor import ServiceMonitor
    
    service_config = {
        'name': 'test-endpoint',
        'url': url,
        'method': 'GET',
        'timeout': timeout,
        'expected_status': 200
    }
    
    monitor = ServiceMonitor()
    result = monitor.check_service_health(service_config)
    
    status_icon = {
        ServiceStatus.HEALTHY: "✅",
        ServiceStatus.UNHEALTHY: "❌",
        ServiceStatus.DEGRADED: "⚠️",
        ServiceStatus.UNKNOWN: "❓"
    }.get(result.status, "❓")
    
    click.echo(f"URL: {url}")
    click.echo(f"Status: {status_icon} {result.status.value}")
    click.echo(f"Response Time: {result.response_time_ms:.0f}ms")
    click.echo(f"HTTP Code: {result.status_code or 'N/A'}")
    
    if result.error_message:
        click.echo(f"Error: {result.error_message}")

if __name__ == '__main__':
    cli()
```

### Step 5: Exercise Tasks

1. **Run basic health checks:**
   ```bash
   uv run python -m service_monitor.cli check
   ```

2. **Test individual endpoints:**
   ```bash
   uv run python -m service_monitor.cli test-endpoint https://api.github.com
   ```

3. **Add custom services:**
   ```bash
   uv run python -m service_monitor.cli add-service "my-api" "https://my-service.com/health"
   ```

4. **Start continuous monitoring:**
   ```bash
   uv run python -m service_monitor.cli monitor --interval 30
   ```

5. **Advanced challenges:**
   - Implement custom authentication for different services
   - Add database storage for historical data
   - Create a web dashboard for visualization
   - Implement alerting to multiple channels (email, PagerDuty)
   - Add SLA tracking and reporting

## Key Takeaways

- HTTP requests are fundamental for API interactions and service monitoring
- Proper error handling and retries ensure robust API integrations
- Authentication methods (API keys, JWT) secure service communications
- Rate limiting prevents overwhelming external services
- Health checks are essential for maintaining service reliability
- Parallel processing improves monitoring performance
- Structured logging and alerting enable proactive incident response

This chapter provides the foundation for building sophisticated monitoring, integration, and automation systems that are core to SRE/DevOps practices.