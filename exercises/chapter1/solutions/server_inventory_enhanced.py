#!/usr/bin/env python3
"""
Enhanced Server Inventory Management System
Demonstrates Python fundamentals for SRE/DevOps with additional features.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class ServerInventory:
    """Enhanced server inventory management system."""
    
    def __init__(self, data_file: str = "servers.json"):
        self.data_file = Path(data_file)
        self.servers = self._load_servers()
        
    def _load_servers(self) -> Dict[str, Dict[str, Any]]:
        """Load servers from JSON file."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading server data: {e}")
                return {}
        return {}
    
    def _save_servers(self):
        """Save servers to JSON file."""
        try:
            # Create directory if it doesn't exist
            self.data_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.data_file, 'w') as f:
                json.dump(self.servers, f, indent=2, default=str)
            print(f"Server data saved to {self.data_file}")
        except IOError as e:
            print(f"Error saving server data: {e}")
    
    def add_server(self, name: str, ip: str, role: str, environment: str = "production") -> bool:
        """Add a new server to inventory."""
        if name in self.servers:
            print(f"Server '{name}' already exists!")
            return False
        
        # Validate IP address format
        if not self._validate_ip(ip):
            print(f"Invalid IP address: {ip}")
            return False
        
        self.servers[name] = {
            "ip": ip,
            "role": role,
            "environment": environment,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "tags": [],
            "metadata": {}
        }
        
        print(f"Server '{name}' added successfully!")
        return True
    
    def remove_server(self, name: str) -> bool:
        """Remove a server from inventory."""
        if name not in self.servers:
            print(f"Server '{name}' not found!")
            return False
        
        del self.servers[name]
        print(f"Server '{name}' removed successfully!")
        return True
    
    def update_server(self, name: str, **kwargs) -> bool:
        """Update server information."""
        if name not in self.servers:
            print(f"Server '{name}' not found!")
            return False
        
        # Update allowed fields
        allowed_fields = ["ip", "role", "environment", "status", "tags", "metadata"]
        updated_fields = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == "ip" and not self._validate_ip(value):
                    print(f"Invalid IP address: {value}")
                    continue
                
                self.servers[name][field] = value
                updated_fields.append(field)
        
        if updated_fields:
            self.servers[name]["last_updated"] = datetime.now().isoformat()
            print(f"Server '{name}' updated: {', '.join(updated_fields)}")
            return True
        else:
            print("No valid fields to update")
            return False
    
    def list_servers(self, filter_by: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """List servers with optional filtering."""
        servers_list = []
        
        for name, info in self.servers.items():
            server_data = {"name": name, **info}
            
            # Apply filters if provided
            if filter_by:
                match = True
                for field, value in filter_by.items():
                    if field not in server_data or str(server_data[field]).lower() != value.lower():
                        match = False
                        break
                if not match:
                    continue
            
            servers_list.append(server_data)
        
        return servers_list
    
    def get_server(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific server."""
        if name in self.servers:
            return {"name": name, **self.servers[name]}
        return None
    
    def search_servers(self, query: str) -> List[Dict[str, Any]]:
        """Search servers by name, IP, or role."""
        results = []
        query_lower = query.lower()
        
        for name, info in self.servers.items():
            if (query_lower in name.lower() or 
                query_lower in info.get("ip", "").lower() or 
                query_lower in info.get("role", "").lower()):
                results.append({"name": name, **info})
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inventory statistics."""
        if not self.servers:
            return {"total": 0}
        
        # Count by role
        roles = {}
        environments = {}
        statuses = {}
        
        for server_info in self.servers.values():
            role = server_info.get("role", "unknown")
            env = server_info.get("environment", "unknown")
            status = server_info.get("status", "unknown")
            
            roles[role] = roles.get(role, 0) + 1
            environments[env] = environments.get(env, 0) + 1
            statuses[status] = statuses.get(status, 0) + 1
        
        return {
            "total": len(self.servers),
            "by_role": roles,
            "by_environment": environments,
            "by_status": statuses
        }
    
    def export_to_csv(self, filename: str):
        """Export server inventory to CSV format."""
        import csv
        
        if not self.servers:
            print("No servers to export!")
            return
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ["name", "ip", "role", "environment", "status", "created_at", "last_updated"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for name, info in self.servers.items():
                    row = {"name": name}
                    row.update({field: info.get(field, "") for field in fieldnames[1:]})
                    writer.writerow(row)
            
            print(f"Server inventory exported to {filename}")
        except IOError as e:
            print(f"Error exporting to CSV: {e}")
    
    def backup_data(self, backup_dir: str = "backups"):
        """Create a backup of server data."""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"servers_backup_{timestamp}.json"
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(self.servers, f, indent=2, default=str)
            print(f"Backup created: {backup_file}")
        except IOError as e:
            print(f"Error creating backup: {e}")
    
    @staticmethod
    def _validate_ip(ip: str) -> bool:
        """Validate IP address format."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def print_table(self, servers: List[Dict[str, Any]], fields: List[str] = None):
        """Print servers in a formatted table."""
        if not servers:
            print("No servers found.")
            return
        
        if fields is None:
            fields = ["name", "ip", "role", "environment", "status"]
        
        # Calculate column widths
        widths = {}
        for field in fields:
            widths[field] = max(len(field), max(len(str(server.get(field, ""))) for server in servers))
        
        # Print header
        header = " | ".join(field.ljust(widths[field]) for field in fields)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for server in servers:
            row = " | ".join(str(server.get(field, "")).ljust(widths[field]) for field in fields)
            print(row)

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Server Inventory Management System")
    parser.add_argument("--data-file", default="servers.json", help="Data file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add server command
    add_parser = subparsers.add_parser("add", help="Add a new server")
    add_parser.add_argument("name", help="Server name")
    add_parser.add_argument("ip", help="Server IP address")
    add_parser.add_argument("role", help="Server role")
    add_parser.add_argument("--environment", default="production", help="Environment")
    
    # Remove server command
    remove_parser = subparsers.add_parser("remove", help="Remove a server")
    remove_parser.add_argument("name", help="Server name to remove")
    
    # List servers command
    list_parser = subparsers.add_parser("list", help="List servers")
    list_parser.add_argument("--role", help="Filter by role")
    list_parser.add_argument("--environment", help="Filter by environment")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    
    # Update server command
    update_parser = subparsers.add_parser("update", help="Update server information")
    update_parser.add_argument("name", help="Server name")
    update_parser.add_argument("--ip", help="New IP address")
    update_parser.add_argument("--role", help="New role")
    update_parser.add_argument("--environment", help="New environment")
    update_parser.add_argument("--status", help="New status")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search servers")
    search_parser.add_argument("query", help="Search query")
    
    # Statistics command
    subparsers.add_parser("stats", help="Show inventory statistics")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export to CSV")
    export_parser.add_argument("filename", help="Output CSV filename")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--dir", default="backups", help="Backup directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize inventory
    inventory = ServerInventory(args.data_file)
    
    # Execute commands
    try:
        if args.command == "add":
            success = inventory.add_server(args.name, args.ip, args.role, args.environment)
            if success:
                inventory._save_servers()
        
        elif args.command == "remove":
            success = inventory.remove_server(args.name)
            if success:
                inventory._save_servers()
        
        elif args.command == "list":
            filter_by = {}
            if args.role:
                filter_by["role"] = args.role
            if args.environment:
                filter_by["environment"] = args.environment
            if args.status:
                filter_by["status"] = args.status
            
            servers = inventory.list_servers(filter_by if filter_by else None)
            
            if args.format == "json":
                print(json.dumps(servers, indent=2, default=str))
            else:
                inventory.print_table(servers)
        
        elif args.command == "update":
            updates = {}
            if args.ip:
                updates["ip"] = args.ip
            if args.role:
                updates["role"] = args.role
            if args.environment:
                updates["environment"] = args.environment
            if args.status:
                updates["status"] = args.status
            
            if updates:
                success = inventory.update_server(args.name, **updates)
                if success:
                    inventory._save_servers()
            else:
                print("No updates specified!")
        
        elif args.command == "search":
            servers = inventory.search_servers(args.query)
            inventory.print_table(servers)
        
        elif args.command == "stats":
            stats = inventory.get_statistics()
            print("Server Inventory Statistics:")
            print(f"Total Servers: {stats['total']}")
            
            if stats['total'] > 0:
                print("\nBy Role:")
                for role, count in stats['by_role'].items():
                    print(f"  {role}: {count}")
                
                print("\nBy Environment:")
                for env, count in stats['by_environment'].items():
                    print(f"  {env}: {count}")
                
                print("\nBy Status:")
                for status, count in stats['by_status'].items():
                    print(f"  {status}: {count}")
        
        elif args.command == "export":
            inventory.export_to_csv(args.filename)
        
        elif args.command == "backup":
            inventory.backup_data(args.dir)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()