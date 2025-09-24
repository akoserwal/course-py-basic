#!/usr/bin/env python3
"""
Exercise 6A.1: Basic Logging
Learn to add logging to your applications for better debugging and monitoring.
"""

import logging
import time
import random
from datetime import datetime

# TODO: Set up basic logging configuration
# Uncomment and complete this configuration:
#
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
#     handlers=[
#         logging.StreamHandler(),  # Print to console
#         logging.FileHandler('exercise_app.log')  # Save to file
#     ]
# )

logger = logging.getLogger(__name__)

def process_server_request(server_name: str, request_type: str = "health_check"):
    """
    Process a server request with logging.
    
    TODO: Add appropriate logging statements throughout this function.
    """
    
    # TODO: Log that we're starting to process the request
    print(f"Processing {request_type} request for {server_name}")
    
    try:
        # Simulate processing time
        processing_time = random.uniform(0.1, 2.0)
        time.sleep(processing_time)
        
        # Simulate random failures
        if random.random() < 0.2:  # 20% chance of failure
            raise Exception(f"Server {server_name} is not responding")
        
        # TODO: Log successful processing
        print(f"Successfully processed {server_name} in {processing_time:.2f} seconds")
        
        return {
            "status": "success",
            "server": server_name,
            "request_type": request_type,
            "processing_time": processing_time
        }
        
    except Exception as e:
        # TODO: Log the error with appropriate level
        print(f"Failed to process {server_name}: {e}")
        
        return {
            "status": "error",
            "server": server_name,
            "request_type": request_type,
            "error": str(e)
        }

def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists with proper logging.
    
    TODO: Add logging to this function.
    """
    import os
    
    # TODO: Log that we're checking the file
    
    try:
        exists = os.path.exists(file_path)
        
        if exists:
            # TODO: Log file found (INFO level)
            pass
        else:
            # TODO: Log file not found (WARNING level)
            pass
        
        return exists
        
    except Exception as e:
        # TODO: Log any errors (ERROR level)
        return False

def demo_different_log_levels():
    """Demonstrate different logging levels."""
    
    print("\nDemonstrating different log levels:")
    print("-" * 40)
    
    # TODO: Add logging statements for each level:
    
    # DEBUG: Very detailed information, usually only used when debugging
    print("This would be a DEBUG message")
    
    # INFO: General information about what's happening  
    print("This would be an INFO message")
    
    # WARNING: Something unexpected happened, but the program is still working
    print("This would be a WARNING message")
    
    # ERROR: Something went wrong, but the program can continue
    print("This would be an ERROR message")
    
    # CRITICAL: Something very serious happened, the program might stop
    print("This would be a CRITICAL message")

def monitor_server_health():
    """Monitor multiple servers and log the results."""
    
    servers = [
        "web-01",
        "web-02", 
        "database-server",
        "cache-server",
        "broken-server"  # This one will likely fail
    ]
    
    print("\nStarting server health monitoring...")
    print("=" * 50)
    
    healthy_servers = 0
    total_servers = len(servers)
    
    for server in servers:
        result = process_server_request(server, "health_check")
        
        if result["status"] == "success":
            healthy_servers += 1
        
        # Small delay between checks
        time.sleep(0.5)
    
    # TODO: Log summary statistics
    health_percentage = (healthy_servers / total_servers) * 100
    
    print(f"\nHealth check summary:")
    print(f"Healthy: {healthy_servers}/{total_servers} ({health_percentage:.1f}%)")
    
    if health_percentage == 100:
        # TODO: Log that all servers are healthy (INFO)
        pass
    elif health_percentage >= 80:
        # TODO: Log that most servers are healthy but some issues exist (WARNING)
        pass
    else:
        # TODO: Log that there are serious health issues (ERROR)
        pass

def main():
    """
    Exercise Instructions:
    
    1. Complete the logging configuration at the top of the file
    2. Add appropriate logging statements to all the TODO sections
    3. Run the script and observe the console output
    4. Check the 'exercise_app.log' file that gets created
    5. Try changing the logging level to DEBUG and see what changes
    
    Tasks to complete:
    1. Set up logging configuration
    2. Add logging to process_server_request()
    3. Add logging to check_file_exists()
    4. Complete demo_different_log_levels()
    5. Add summary logging to monitor_server_health()
    
    Bonus challenges:
    1. Add a timestamp to log messages
    2. Create separate log files for different log levels
    3. Add logging for function entry and exit (DEBUG level)
    4. Log performance metrics (how long functions take)
    """
    
    print("Exercise 6A.1: Basic Logging")
    print("=" * 40)
    
    # Demo different log levels
    demo_different_log_levels()
    
    # Test file checking
    print("\nTesting file existence checking:")
    check_file_exists("/etc/passwd")  # Likely exists on Unix systems
    check_file_exists("/nonexistent/file.txt")  # Definitely doesn't exist
    
    # Monitor server health
    monitor_server_health()
    
    print("\nExercise completed!")
    print("Check the 'exercise_app.log' file to see the logged messages.")

# Challenge function - uncomment to try advanced logging
def challenge_performance_logging():
    """Challenge: Add performance logging to track function execution time."""
    
    def log_performance(func):
        """Decorator to log function performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # TODO: Log function start (DEBUG level)
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                # TODO: Log successful execution with timing (DEBUG level)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                # TODO: Log failed execution with timing (ERROR level)
                
                raise
        
        return wrapper
    
    # Example usage:
    @log_performance
    def slow_operation():
        time.sleep(1)
        return "Operation completed"
    
    print("\nTesting performance logging:")
    result = slow_operation()
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
    
    # Uncomment to try the performance logging challenge:
    # challenge_performance_logging()