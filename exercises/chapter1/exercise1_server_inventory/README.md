# Exercise 1: Server Inventory Management System

## Objective

Build a server inventory management system that tracks servers, their specifications, and status. This exercise focuses on Python data structures, functions, and basic object-oriented programming.

## Requirements

Create a system that can:

1. **Add servers** with the following information:
   - Name (unique identifier)
   - IP address
   - Operating system
   - CPU cores
   - Memory (GB)
   - Disk space (GB)
   - Status (online/offline/maintenance)
   - Tags (list of strings)

2. **Update server information** including status changes

3. **Search and filter servers** by:
   - Status
   - Operating system
   - Tags
   - Resource specifications (CPU, memory, disk)

4. **Generate reports**:
   - Total servers by status
   - Resource utilization summary
   - Servers requiring maintenance

5. **Data persistence**: Save/load inventory from JSON file

## Starter Code

Use the provided `server_inventory_starter.py` as your starting point.

## Implementation Tasks

### Task 1: Server Class
Create a `Server` class with:
- Constructor that accepts all server properties
- Methods to update status and specifications
- String representation for display
- Validation for IP addresses and positive numbers

### Task 2: Inventory Manager
Create a `ServerInventory` class with:
- Methods to add, remove, and update servers
- Search functionality with multiple filters
- Report generation methods
- JSON serialization/deserialization

### Task 3: CLI Interface
Create a command-line interface that allows:
- Interactive server management
- Bulk operations from CSV files
- Report generation and export

## Test Cases

Your implementation should pass all tests in `test_server_inventory.py`:

```bash
python -m pytest test_server_inventory.py -v
```

## Sample Data

Use the provided `sample_servers.json` for testing:

```json
[
  {
    "name": "web-server-01",
    "ip": "192.168.1.10",
    "os": "Ubuntu 22.04",
    "cpu_cores": 4,
    "memory_gb": 16,
    "disk_gb": 500,
    "status": "online",
    "tags": ["web", "production", "frontend"]
  },
  {
    "name": "db-server-01", 
    "ip": "192.168.1.20",
    "os": "CentOS 8",
    "cpu_cores": 8,
    "memory_gb": 32,
    "disk_gb": 1000,
    "status": "online",
    "tags": ["database", "production", "mysql"]
  },
  {
    "name": "test-server-01",
    "ip": "192.168.1.100",
    "os": "Ubuntu 22.04", 
    "cpu_cores": 2,
    "memory_gb": 8,
    "disk_gb": 250,
    "status": "offline",
    "tags": ["testing", "development"]
  }
]
```

## Expected Output

When you run `python server_inventory.py`, you should see:

```
=== Server Inventory Management System ===

1. Add Server
2. List Servers
3. Update Server
4. Search Servers
5. Generate Reports
6. Load from File
7. Save to File
8. Exit

Enter your choice: 2

=== Server Inventory ===
Name: web-server-01 | IP: 192.168.1.10 | Status: online | OS: Ubuntu 22.04
  Resources: 4 CPU cores, 16 GB RAM, 500 GB disk
  Tags: web, production, frontend

Name: db-server-01 | IP: 192.168.1.20 | Status: online | OS: CentOS 8
  Resources: 8 CPU cores, 32 GB RAM, 1000 GB disk
  Tags: database, production, mysql

Name: test-server-01 | IP: 192.168.1.100 | Status: offline | OS: Ubuntu 22.04
  Resources: 2 CPU cores, 8 GB RAM, 250 GB disk
  Tags: testing, development

Total servers: 3
```

## Bonus Challenges

1. **Resource Monitoring**: Add CPU/memory usage tracking
2. **Network Scanning**: Implement ping functionality to auto-update status
3. **Reporting**: Generate detailed reports with charts using matplotlib
4. **Configuration Management**: Track installed software packages
5. **Alert System**: Notify when servers go offline or need maintenance

## Files to Submit

- `server_inventory.py` - Your main implementation
- `test_results.txt` - Output of running all tests
- `sample_output.txt` - Example of your program running