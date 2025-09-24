#!/usr/bin/env python3
"""
Server Inventory Management System - Complete Solution

This is the complete implementation of the server inventory system.
Study this code to understand the full solution.
"""

import json
import csv
import ipaddress
from typing import List, Dict, Any, Optional
from datetime import datetime


class Server:
    """Represents a server in the inventory."""
    
    VALID_STATUSES = {'online', 'offline', 'maintenance'}
    
    def __init__(self, name: str, ip: str, os: str, cpu_cores: int, 
                 memory_gb: int, disk_gb: int, status: str = "offline", 
                 tags: List[str] = None):
        """Initialize a server instance with validation."""
        self.name = self._validate_name(name)
        self.ip = self._validate_ip(ip)
        self.os = os
        self.cpu_cores = self._validate_positive_int(cpu_cores, "CPU cores")
        self.memory_gb = self._validate_positive_int(memory_gb, "Memory")
        self.disk_gb = self._validate_positive_int(disk_gb, "Disk space")
        self.status = self._validate_status(status)
        self.tags = tags if tags is not None else []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def _validate_name(self, name: str) -> str:
        """Validate server name."""
        if not name or not isinstance(name, str):
            raise ValueError("Server name must be a non-empty string")
        return name.strip()
    
    def _validate_ip(self, ip: str) -> str:
        """Validate IP address format."""
        try:
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            raise ValueError(f"Invalid IP address format: {ip}")
    
    def _validate_positive_int(self, value: int, field_name: str) -> int:
        """Validate positive integer values."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return value
    
    def _validate_status(self, status: str) -> str:
        """Validate server status."""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Status must be one of: {', '.join(self.VALID_STATUSES)}")
        return status
    
    def update_status(self, new_status: str):
        """Update server status."""
        self.status = self._validate_status(new_status)
        self.updated_at = datetime.now()
    
    def update_resources(self, cpu_cores: int = None, memory_gb: int = None, disk_gb: int = None):
        """Update server resource specifications."""
        if cpu_cores is not None:
            self.cpu_cores = self._validate_positive_int(cpu_cores, "CPU cores")
        if memory_gb is not None:
            self.memory_gb = self._validate_positive_int(memory_gb, "Memory")
        if disk_gb is not None:
            self.disk_gb = self._validate_positive_int(disk_gb, "Disk space")
        
        if any(param is not None for param in [cpu_cores, memory_gb, disk_gb]):
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the server (avoid duplicates)."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def remove_tag(self, tag: str):
        """Remove a tag from the server."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()
    
    def matches_filter(self, **filters) -> bool:
        """Check if server matches given filters."""
        for key, value in filters.items():
            if key == 'status' and self.status != value:
                return False
            elif key == 'os' and value.lower() not in self.os.lower():
                return False
            elif key == 'tags':
                # Check if any of the required tags match
                if isinstance(value, list):
                    if not any(tag in self.tags for tag in value):
                        return False
                elif value not in self.tags:
                    return False
            elif key == 'min_cpu' and self.cpu_cores < value:
                return False
            elif key == 'min_memory' and self.memory_gb < value:
                return False
            elif key == 'min_disk' and self.disk_gb < value:
                return False
            elif key == 'max_cpu' and self.cpu_cores > value:
                return False
            elif key == 'max_memory' and self.memory_gb > value:
                return False
            elif key == 'max_disk' and self.disk_gb > value:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert server to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'ip': self.ip,
            'os': self.os,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'disk_gb': self.disk_gb,
            'status': self.status,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Server':
        """Create server instance from dictionary."""
        server = cls(
            name=data['name'],
            ip=data['ip'],
            os=data['os'],
            cpu_cores=data['cpu_cores'],
            memory_gb=data['memory_gb'],
            disk_gb=data['disk_gb'],
            status=data.get('status', 'offline'),
            tags=data.get('tags', [])
        )
        
        # Restore timestamps if available
        if 'created_at' in data:
            server.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            server.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return server
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        tag_str = ', '.join(self.tags) if self.tags else 'None'
        return (f"Name: {self.name} | IP: {self.ip} | Status: {self.status} | OS: {self.os}\n"
                f"  Resources: {self.cpu_cores} CPU cores, {self.memory_gb} GB RAM, {self.disk_gb} GB disk\n"
                f"  Tags: {tag_str}")
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"Server(name='{self.name}', ip='{self.ip}', status='{self.status}', "
                f"cpu_cores={self.cpu_cores}, memory_gb={self.memory_gb}, disk_gb={self.disk_gb})")


class ServerInventory:
    """Manages a collection of servers."""
    
    def __init__(self):
        """Initialize empty server inventory."""
        self.servers: Dict[str, Server] = {}
        self.created_at = datetime.now()
    
    def add_server(self, server: Server) -> bool:
        """Add a server to the inventory."""
        if server.name in self.servers:
            return False  # Duplicate name
        
        self.servers[server.name] = server
        return True
    
    def remove_server(self, name: str) -> bool:
        """Remove a server from the inventory."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False
    
    def get_server(self, name: str) -> Optional[Server]:
        """Get a server by name."""
        return self.servers.get(name)
    
    def update_server_status(self, name: str, status: str) -> bool:
        """Update server status."""
        server = self.get_server(name)
        if server:
            server.update_status(status)
            return True
        return False
    
    def list_servers(self, **filters) -> List[Server]:
        """List servers with optional filtering."""
        if not filters:
            return list(self.servers.values())
        
        return [server for server in self.servers.values() 
                if server.matches_filter(**filters)]
    
    def search_servers(self, **criteria) -> List[Server]:
        """Search servers by multiple criteria."""
        return self.list_servers(**criteria)
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate a status report of all servers."""
        status_counts = {'online': 0, 'offline': 0, 'maintenance': 0}
        total_servers = len(self.servers)
        
        for server in self.servers.values():
            status_counts[server.status] += 1
        
        return {
            'total_servers': total_servers,
            'status_breakdown': status_counts,
            'online_percentage': (status_counts['online'] / total_servers * 100) if total_servers > 0 else 0,
            'offline_count': status_counts['offline'],
            'maintenance_count': status_counts['maintenance'],
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_resource_report(self) -> Dict[str, Any]:
        """Generate a resource utilization report."""
        if not self.servers:
            return {
                'total_servers': 0,
                'total_resources': {'cpu_cores': 0, 'memory_gb': 0, 'disk_gb': 0},
                'average_resources': {'cpu_cores': 0, 'memory_gb': 0, 'disk_gb': 0}
            }
        
        total_cpu = sum(server.cpu_cores for server in self.servers.values())
        total_memory = sum(server.memory_gb for server in self.servers.values())
        total_disk = sum(server.disk_gb for server in self.servers.values())
        
        server_count = len(self.servers)
        
        return {
            'total_servers': server_count,
            'total_resources': {
                'cpu_cores': total_cpu,
                'memory_gb': total_memory,
                'disk_gb': total_disk
            },
            'average_resources': {
                'cpu_cores': round(total_cpu / server_count, 2),
                'memory_gb': round(total_memory / server_count, 2),
                'disk_gb': round(total_disk / server_count, 2)
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def get_servers_needing_maintenance(self) -> List[Server]:
        """Get servers that need maintenance."""
        return [server for server in self.servers.values() 
                if server.status == 'maintenance']
    
    def save_to_file(self, filename: str) -> bool:
        """Save inventory to JSON file."""
        try:
            data = {
                'servers': [server.to_dict() for server in self.servers.values()],
                'inventory_created_at': self.created_at.isoformat(),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving to file: {e}")
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """Load inventory from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Clear existing servers
            self.servers.clear()
            
            # Load servers
            for server_data in data.get('servers', []):
                server = Server.from_dict(server_data)
                self.servers[server.name] = server
            
            # Restore inventory creation time if available
            if 'inventory_created_at' in data:
                self.created_at = datetime.fromisoformat(data['inventory_created_at'])
            
            return True
        except Exception as e:
            print(f"Error loading from file: {e}")
            return False
    
    def import_from_csv(self, filename: str) -> int:
        """Import servers from CSV file."""
        imported_count = 0
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse tags (assume comma-separated in CSV)
                        tags = [tag.strip() for tag in row.get('tags', '').split(',') if tag.strip()]
                        
                        server = Server(
                            name=row['name'],
                            ip=row['ip'],
                            os=row['os'],
                            cpu_cores=int(row['cpu_cores']),
                            memory_gb=int(row['memory_gb']),
                            disk_gb=int(row['disk_gb']),
                            status=row.get('status', 'offline'),
                            tags=tags
                        )
                        
                        if self.add_server(server):
                            imported_count += 1
                        else:
                            print(f"Skipped duplicate server: {server.name}")
                    
                    except Exception as e:
                        print(f"Error importing row {row}: {e}")
            
            return imported_count
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 0
    
    def export_to_csv(self, filename: str) -> bool:
        """Export servers to CSV file."""
        try:
            with open(filename, 'w', newline='') as f:
                fieldnames = ['name', 'ip', 'os', 'cpu_cores', 'memory_gb', 'disk_gb', 'status', 'tags']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                
                for server in self.servers.values():
                    row = {
                        'name': server.name,
                        'ip': server.ip,
                        'os': server.os,
                        'cpu_cores': server.cpu_cores,
                        'memory_gb': server.memory_gb,
                        'disk_gb': server.disk_gb,
                        'status': server.status,
                        'tags': ', '.join(server.tags)
                    }
                    writer.writerow(row)
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


def print_servers(servers: List[Server]):
    """Helper function to print a list of servers."""
    if not servers:
        print("No servers found.")
        return
    
    print(f"\n=== Found {len(servers)} server(s) ===")
    for server in servers:
        print(server)
        print()


def main():
    """Main CLI interface for the server inventory system."""
    inventory = ServerInventory()
    
    # Load sample data if available
    try:
        if inventory.load_from_file("sample_servers.json"):
            print("Loaded sample data from sample_servers.json")
    except:
        pass  # No sample data available
    
    while True:
        print("\n=== Server Inventory Management System ===")
        print("1. Add Server")
        print("2. List Servers")
        print("3. Update Server")
        print("4. Search Servers")
        print("5. Generate Reports")
        print("6. Load from File")
        print("7. Save to File")
        print("8. Import from CSV")
        print("9. Export to CSV")
        print("10. Exit")
        
        choice = input("\nEnter your choice (1-10): ").strip()
        
        if choice == '1':
            # Add Server
            try:
                print("\n=== Add New Server ===")
                name = input("Server name: ").strip()
                ip = input("IP address: ").strip()
                os = input("Operating system: ").strip()
                cpu_cores = int(input("CPU cores: "))
                memory_gb = int(input("Memory (GB): "))
                disk_gb = int(input("Disk space (GB): "))
                status = input("Status (online/offline/maintenance) [offline]: ").strip() or "offline"
                tags_input = input("Tags (comma-separated): ").strip()
                tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                
                server = Server(name, ip, os, cpu_cores, memory_gb, disk_gb, status, tags)
                
                if inventory.add_server(server):
                    print(f"✓ Server '{name}' added successfully!")
                else:
                    print(f"✗ Server '{name}' already exists!")
            
            except ValueError as e:
                print(f"✗ Error: {e}")
            except Exception as e:
                print(f"✗ Unexpected error: {e}")
        
        elif choice == '2':
            # List Servers
            print("\n=== Server Inventory ===")
            servers = inventory.list_servers()
            print_servers(servers)
            print(f"Total servers: {len(servers)}")
        
        elif choice == '3':
            # Update Server
            print("\n=== Update Server ===")
            name = input("Enter server name to update: ").strip()
            server = inventory.get_server(name)
            
            if not server:
                print(f"✗ Server '{name}' not found!")
                continue
            
            print(f"Current server info:\n{server}\n")
            
            print("What would you like to update?")
            print("1. Status")
            print("2. Resources")
            print("3. Add Tag")
            print("4. Remove Tag")
            
            update_choice = input("Enter choice (1-4): ").strip()
            
            try:
                if update_choice == '1':
                    new_status = input("New status (online/offline/maintenance): ").strip()
                    server.update_status(new_status)
                    print("✓ Status updated!")
                
                elif update_choice == '2':
                    cpu_input = input(f"CPU cores [{server.cpu_cores}]: ").strip()
                    memory_input = input(f"Memory GB [{server.memory_gb}]: ").strip()
                    disk_input = input(f"Disk GB [{server.disk_gb}]: ").strip()
                    
                    cpu = int(cpu_input) if cpu_input else None
                    memory = int(memory_input) if memory_input else None
                    disk = int(disk_input) if disk_input else None
                    
                    server.update_resources(cpu, memory, disk)
                    print("✓ Resources updated!")
                
                elif update_choice == '3':
                    tag = input("Tag to add: ").strip()
                    server.add_tag(tag)
                    print("✓ Tag added!")
                
                elif update_choice == '4':
                    tag = input("Tag to remove: ").strip()
                    server.remove_tag(tag)
                    print("✓ Tag removed!")
                
                else:
                    print("✗ Invalid choice!")
            
            except ValueError as e:
                print(f"✗ Error: {e}")
        
        elif choice == '4':
            # Search Servers
            print("\n=== Search Servers ===")
            print("Available filters:")
            print("1. By status")
            print("2. By OS")
            print("3. By tags")
            print("4. By minimum resources")
            print("5. Show all")
            
            search_choice = input("Enter choice (1-5): ").strip()
            filters = {}
            
            try:
                if search_choice == '1':
                    status = input("Status (online/offline/maintenance): ").strip()
                    filters['status'] = status
                
                elif search_choice == '2':
                    os_filter = input("OS (partial match): ").strip()
                    filters['os'] = os_filter
                
                elif search_choice == '3':
                    tags_input = input("Tags (comma-separated): ").strip()
                    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                    filters['tags'] = tags
                
                elif search_choice == '4':
                    min_cpu = input("Minimum CPU cores: ").strip()
                    min_memory = input("Minimum memory GB: ").strip()
                    min_disk = input("Minimum disk GB: ").strip()
                    
                    if min_cpu:
                        filters['min_cpu'] = int(min_cpu)
                    if min_memory:
                        filters['min_memory'] = int(min_memory)
                    if min_disk:
                        filters['min_disk'] = int(min_disk)
                
                elif search_choice == '5':
                    pass  # No filters
                
                else:
                    print("✗ Invalid choice!")
                    continue
                
                servers = inventory.search_servers(**filters)
                print_servers(servers)
            
            except ValueError as e:
                print(f"✗ Error: {e}")
        
        elif choice == '5':
            # Generate Reports
            print("\n=== Generate Reports ===")
            print("1. Status Report")
            print("2. Resource Report")
            print("3. Maintenance Report")
            
            report_choice = input("Enter choice (1-3): ").strip()
            
            if report_choice == '1':
                report = inventory.generate_status_report()
                print("\n=== Status Report ===")
                print(f"Total servers: {report['total_servers']}")
                print(f"Online: {report['status_breakdown']['online']} ({report['online_percentage']:.1f}%)")
                print(f"Offline: {report['status_breakdown']['offline']}")
                print(f"Maintenance: {report['status_breakdown']['maintenance']}")
                print(f"Generated at: {report['generated_at']}")
            
            elif report_choice == '2':
                report = inventory.generate_resource_report()
                print("\n=== Resource Report ===")
                print(f"Total servers: {report['total_servers']}")
                print("\nTotal Resources:")
                print(f"  CPU cores: {report['total_resources']['cpu_cores']}")
                print(f"  Memory: {report['total_resources']['memory_gb']} GB")
                print(f"  Disk: {report['total_resources']['disk_gb']} GB")
                print("\nAverage per server:")
                print(f"  CPU cores: {report['average_resources']['cpu_cores']}")
                print(f"  Memory: {report['average_resources']['memory_gb']} GB")
                print(f"  Disk: {report['average_resources']['disk_gb']} GB")
                print(f"Generated at: {report['generated_at']}")
            
            elif report_choice == '3':
                servers = inventory.get_servers_needing_maintenance()
                print("\n=== Maintenance Report ===")
                print_servers(servers)
            
            else:
                print("✗ Invalid choice!")
        
        elif choice == '6':
            # Load from File
            filename = input("Enter filename to load from: ").strip()
            if inventory.load_from_file(filename):
                print(f"✓ Successfully loaded from {filename}")
            else:
                print(f"✗ Failed to load from {filename}")
        
        elif choice == '7':
            # Save to File
            filename = input("Enter filename to save to: ").strip()
            if inventory.save_to_file(filename):
                print(f"✓ Successfully saved to {filename}")
            else:
                print(f"✗ Failed to save to {filename}")
        
        elif choice == '8':
            # Import from CSV
            filename = input("Enter CSV filename to import from: ").strip()
            count = inventory.import_from_csv(filename)
            print(f"✓ Imported {count} servers from {filename}")
        
        elif choice == '9':
            # Export to CSV
            filename = input("Enter CSV filename to export to: ").strip()
            if inventory.export_to_csv(filename):
                print(f"✓ Successfully exported to {filename}")
            else:
                print(f"✗ Failed to export to {filename}")
        
        elif choice == '10':
            print("Goodbye!")
            break
        
        else:
            print("✗ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()