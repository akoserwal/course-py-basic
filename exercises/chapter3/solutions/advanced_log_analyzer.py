#!/usr/bin/env python3
"""
Advanced Log Analyzer for DevOps and SRE
Comprehensive log analysis tool with filtering, aggregation, and alerting.
"""

import re
import sys
import json
import gzip
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import yaml

@dataclass
class LogEntry:
    """Represents a single log entry."""
    timestamp: datetime
    level: str
    message: str
    source: str
    raw_line: str
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    total_entries: int
    entries_by_level: Dict[str, int]
    error_patterns: Dict[str, int]
    time_range: Tuple[datetime, datetime]
    top_errors: List[Dict[str, Any]]
    hourly_distribution: Dict[str, int]
    source_analysis: Dict[str, int]

class LogPatterns:
    """Common log patterns for different formats."""
    
    # Apache/Nginx access log
    APACHE_COMMON = re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\S+)'
    )
    
    # Standard application log format
    APP_LOG = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>\w+)\] (?P<message>.*)'
    )
    
    # Syslog format
    SYSLOG = re.compile(
        r'(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<process>\S+):\s+(?P<message>.*)'
    )
    
    # JSON log format
    JSON_LOG = re.compile(r'^\s*\{.*\}\s*$')
    
    # Kubernetes log format
    K8S_LOG = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(?P<stream>\w+)\s+(?P<flags>\w+)\s+(?P<message>.*)'
    )

class LogAnalyzer:
    """Advanced log analyzer with multiple format support."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.error_patterns = self._compile_error_patterns()
        self.entries: List[LogEntry] = []
        
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'error_patterns': [
                r'(?i)error',
                r'(?i)exception',
                r'(?i)failed',
                r'(?i)timeout',
                r'(?i)connection.*refused',
                r'(?i)out of memory',
                r'(?i)segmentation fault',
                r'(?i)stack trace',
                r'HTTP/\d\.\d" 5\d{2}',  # 5xx HTTP errors
                r'HTTP/\d\.\d" 4\d{2}',  # 4xx HTTP errors
            ],
            'warning_patterns': [
                r'(?i)warning',
                r'(?i)deprecated',
                r'(?i)retry',
                r'(?i)slow',
            ],
            'critical_patterns': [
                r'(?i)critical',
                r'(?i)fatal',
                r'(?i)panic',
                r'(?i)disk.*full',
                r'(?i)service.*down',
            ],
            'time_formats': [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%d/%b/%Y:%H:%M:%S %z',
                '%b %d %H:%M:%S',
            ],
            'ignore_patterns': [
                r'DEBUG',
                r'healthcheck',
            ]
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _compile_error_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for error detection."""
        patterns = {}
        for level in ['error_patterns', 'warning_patterns', 'critical_patterns']:
            patterns[level] = [
                re.compile(pattern) for pattern in self.config.get(level, [])
            ]
        return patterns
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        # Clean timestamp string
        timestamp_str = timestamp_str.strip('[]')
        
        for fmt in self.config['time_formats']:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        # Try parsing with dateutil as fallback
        try:
            from dateutil import parser
            return parser.parse(timestamp_str)
        except:
            pass
        
        return None
    
    def _detect_log_format(self, line: str) -> str:
        """Detect the log format of a line."""
        if LogPatterns.JSON_LOG.match(line):
            return 'json'
        elif LogPatterns.APACHE_COMMON.match(line):
            return 'apache'
        elif LogPatterns.K8S_LOG.match(line):
            return 'kubernetes'
        elif LogPatterns.SYSLOG.match(line):
            return 'syslog'
        elif LogPatterns.APP_LOG.match(line):
            return 'application'
        else:
            return 'unknown'
    
    def _parse_log_line(self, line: str, source: str) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry object."""
        line = line.strip()
        if not line:
            return None
        
        # Check ignore patterns
        for pattern in self.config.get('ignore_patterns', []):
            if re.search(pattern, line, re.IGNORECASE):
                return None
        
        log_format = self._detect_log_format(line)
        timestamp = None
        level = 'INFO'
        message = line
        metadata = {'format': log_format}
        
        try:
            if log_format == 'json':
                data = json.loads(line)
                timestamp = self._parse_timestamp(data.get('timestamp', data.get('time', '')))
                level = data.get('level', data.get('severity', 'INFO')).upper()
                message = data.get('message', data.get('msg', line))
                metadata.update(data)
            
            elif log_format == 'apache':
                match = LogPatterns.APACHE_COMMON.match(line)
                if match:
                    groups = match.groupdict()
                    timestamp = self._parse_timestamp(groups['timestamp'])
                    status_code = int(groups['status'])
                    if status_code >= 500:
                        level = 'ERROR'
                    elif status_code >= 400:
                        level = 'WARNING'
                    else:
                        level = 'INFO'
                    message = f"{groups['method']} {groups['url']} - {status_code}"
                    metadata.update(groups)
            
            elif log_format == 'kubernetes':
                match = LogPatterns.K8S_LOG.match(line)
                if match:
                    groups = match.groupdict()
                    timestamp = self._parse_timestamp(groups['timestamp'])
                    message = groups['message']
                    metadata.update(groups)
            
            elif log_format == 'syslog':
                match = LogPatterns.SYSLOG.match(line)
                if match:
                    groups = match.groupdict()
                    # Construct timestamp (syslog doesn't include year)
                    current_year = datetime.now().year
                    timestamp_str = f"{current_year} {groups['month']} {groups['day']} {groups['time']}"
                    timestamp = self._parse_timestamp(timestamp_str)
                    message = groups['message']
                    metadata.update(groups)
            
            elif log_format == 'application':
                match = LogPatterns.APP_LOG.match(line)
                if match:
                    groups = match.groupdict()
                    timestamp = self._parse_timestamp(groups['timestamp'])
                    level = groups['level'].upper()
                    message = groups['message']
                    metadata.update(groups)
        
        except Exception as e:
            # Fallback parsing
            pass
        
        # Determine log level from message content if not already set
        if level == 'INFO':
            for pattern_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if pattern.search(line):
                        if 'critical' in pattern_type:
                            level = 'CRITICAL'
                        elif 'error' in pattern_type:
                            level = 'ERROR'
                        elif 'warning' in pattern_type:
                            level = 'WARNING'
                        break
                if level != 'INFO':
                    break
        
        # Use current time if timestamp couldn't be parsed
        if timestamp is None:
            timestamp = datetime.now()
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            source=source,
            raw_line=line,
            metadata=metadata
        )
    
    def parse_files(self, file_paths: List[str]) -> None:
        """Parse multiple log files."""
        self.entries.clear()
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"Warning: File {file_path} does not exist")
                continue
            
            print(f"Parsing {file_path}...")
            
            try:
                if path.suffix == '.gz':
                    file_obj = gzip.open(path, 'rt', encoding='utf-8', errors='ignore')
                else:
                    file_obj = open(path, 'r', encoding='utf-8', errors='ignore')
                
                with file_obj:
                    for line_num, line in enumerate(file_obj, 1):
                        entry = self._parse_log_line(line, str(path))
                        if entry:
                            self.entries.append(entry)
                        
                        # Progress indicator for large files
                        if line_num % 10000 == 0:
                            print(f"  Processed {line_num:,} lines...")
            
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
        
        print(f"Parsed {len(self.entries):,} log entries from {len(file_paths)} files")
    
    def filter_entries(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      levels: Optional[List[str]] = None,
                      sources: Optional[List[str]] = None,
                      pattern: Optional[str] = None) -> List[LogEntry]:
        """Filter log entries based on criteria."""
        filtered = self.entries
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        if levels:
            levels_upper = [level.upper() for level in levels]
            filtered = [e for e in filtered if e.level in levels_upper]
        
        if sources:
            filtered = [e for e in filtered if any(src in e.source for src in sources)]
        
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            filtered = [e for e in filtered if regex.search(e.message) or regex.search(e.raw_line)]
        
        return filtered
    
    def analyze(self, entries: Optional[List[LogEntry]] = None) -> AnalysisResult:
        """Perform comprehensive analysis on log entries."""
        if entries is None:
            entries = self.entries
        
        if not entries:
            return AnalysisResult(
                total_entries=0,
                entries_by_level={},
                error_patterns={},
                time_range=(datetime.now(), datetime.now()),
                top_errors=[],
                hourly_distribution={},
                source_analysis={}
            )
        
        # Basic statistics
        total_entries = len(entries)
        entries_by_level = Counter(entry.level for entry in entries)
        source_analysis = Counter(entry.source for entry in entries)
        
        # Time range
        timestamps = [entry.timestamp for entry in entries]
        time_range = (min(timestamps), max(timestamps))
        
        # Hourly distribution
        hourly_distribution = defaultdict(int)
        for entry in entries:
            hour_key = entry.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_distribution[hour_key] += 1
        
        # Error pattern analysis
        error_patterns = defaultdict(int)
        error_entries = [e for e in entries if e.level in ['ERROR', 'CRITICAL']]
        
        for entry in error_entries:
            # Extract common error patterns
            message = entry.message.lower()
            
            # Common patterns
            if 'connection' in message and ('refused' in message or 'timeout' in message):
                error_patterns['Connection Issues'] += 1
            elif 'memory' in message and ('out of' in message or 'limit' in message):
                error_patterns['Memory Issues'] += 1
            elif 'disk' in message and ('full' in message or 'space' in message):
                error_patterns['Disk Space Issues'] += 1
            elif any(word in message for word in ['exception', 'traceback', 'stacktrace']):
                error_patterns['Application Exceptions'] += 1
            elif 'permission' in message and 'denied' in message:
                error_patterns['Permission Errors'] += 1
            elif any(word in message for word in ['timeout', 'deadline']):
                error_patterns['Timeout Errors'] += 1
            else:
                error_patterns['Other Errors'] += 1
        
        # Top errors by frequency
        error_messages = [entry.message for entry in error_entries]
        error_counter = Counter(error_messages)
        top_errors = [
            {'message': msg, 'count': count, 'percentage': (count / len(error_entries)) * 100}
            for msg, count in error_counter.most_common(10)
        ] if error_entries else []
        
        return AnalysisResult(
            total_entries=total_entries,
            entries_by_level=dict(entries_by_level),
            error_patterns=dict(error_patterns),
            time_range=time_range,
            top_errors=top_errors,
            hourly_distribution=dict(hourly_distribution),
            source_analysis=dict(source_analysis)
        )
    
    def generate_report(self, analysis: AnalysisResult, output_format: str = 'text') -> str:
        """Generate analysis report in specified format."""
        if output_format == 'json':
            return json.dumps(asdict(analysis), indent=2, default=str)
        
        elif output_format == 'html':
            return self._generate_html_report(analysis)
        
        else:  # text format
            return self._generate_text_report(analysis)
    
    def _generate_text_report(self, analysis: AnalysisResult) -> str:
        """Generate text-based report."""
        report = []
        report.append("=" * 60)
        report.append("LOG ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Time Range: {analysis.time_range[0]} to {analysis.time_range[1]}")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Entries: {analysis.total_entries:,}")
        
        if analysis.entries_by_level:
            report.append("\nEntries by Level:")
            for level, count in sorted(analysis.entries_by_level.items()):
                percentage = (count / analysis.total_entries) * 100
                report.append(f"  {level}: {count:,} ({percentage:.1f}%)")
        
        # Error Analysis
        if analysis.error_patterns:
            report.append("\nERROR ANALYSIS")
            report.append("-" * 20)
            for pattern, count in sorted(analysis.error_patterns.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {pattern}: {count:,}")
        
        # Top Errors
        if analysis.top_errors:
            report.append("\nTOP ERRORS")
            report.append("-" * 20)
            for i, error in enumerate(analysis.top_errors[:5], 1):
                report.append(f"{i}. {error['message'][:80]}...")
                report.append(f"   Count: {error['count']}, Percentage: {error['percentage']:.1f}%")
                report.append("")
        
        # Source Analysis
        if analysis.source_analysis:
            report.append("SOURCE ANALYSIS")
            report.append("-" * 20)
            for source, count in sorted(analysis.source_analysis.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / analysis.total_entries) * 100
                report.append(f"  {Path(source).name}: {count:,} ({percentage:.1f}%)")
        
        # Hourly Distribution (show last 24 hours)
        if analysis.hourly_distribution:
            report.append("\nHOURLY DISTRIBUTION (Last 24 Hours)")
            report.append("-" * 40)
            
            # Sort by timestamp and show last 24 hours
            sorted_hours = sorted(analysis.hourly_distribution.items())[-24:]
            for hour, count in sorted_hours:
                report.append(f"  {hour}: {count:,}")
        
        return "\n".join(report)
    
    def _generate_html_report(self, analysis: AnalysisResult) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Log Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .error { color: #d32f2f; }
                .warning { color: #f57c00; }
                .critical { color: #b71c1c; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Log Analysis Report</h1>
                <p>Generated: {timestamp}</p>
                <p>Time Range: {start_time} to {end_time}</p>
            </div>
        """.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            start_time=analysis.time_range[0],
            end_time=analysis.time_range[1]
        )
        
        # Add summary metrics
        html += '<div class="section"><h2>Summary</h2>'
        html += f'<div class="metric"><strong>Total Entries:</strong> {analysis.total_entries:,}</div>'
        
        for level, count in analysis.entries_by_level.items():
            percentage = (count / analysis.total_entries) * 100
            css_class = level.lower() if level in ['ERROR', 'WARNING', 'CRITICAL'] else ''
            html += f'<div class="metric {css_class}"><strong>{level}:</strong> {count:,} ({percentage:.1f}%)</div>'
        
        html += '</div>'
        
        # Add error patterns table
        if analysis.error_patterns:
            html += '<div class="section"><h2>Error Patterns</h2><table>'
            html += '<tr><th>Pattern</th><th>Count</th></tr>'
            for pattern, count in sorted(analysis.error_patterns.items(), key=lambda x: x[1], reverse=True):
                html += f'<tr><td>{pattern}</td><td>{count:,}</td></tr>'
            html += '</table></div>'
        
        html += '</body></html>'
        return html
    
    def export_filtered_logs(self, entries: List[LogEntry], filename: str, format: str = 'json'):
        """Export filtered log entries to file."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            data = [asdict(entry) for entry in entries]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'level', 'source', 'message'])
                for entry in entries:
                    writer.writerow([entry.timestamp, entry.level, entry.source, entry.message])
        
        elif format == 'text':
            with open(filename, 'w') as f:
                for entry in entries:
                    f.write(f"{entry.timestamp} [{entry.level}] {entry.message}\n")
        
        print(f"Exported {len(entries):,} entries to {filename}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Advanced Log Analyzer for DevOps")
    parser.add_argument("files", nargs="+", help="Log files to analyze")
    parser.add_argument("--config", help="Configuration file (JSON or YAML)")
    parser.add_argument("--start-time", help="Start time filter (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", help="End time filter (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--levels", nargs="+", help="Log levels to include")
    parser.add_argument("--pattern", help="Regex pattern to search for")
    parser.add_argument("--sources", nargs="+", help="Source files to include")
    parser.add_argument("--output-format", choices=["text", "json", "html"], default="text", help="Output format")
    parser.add_argument("--export", help="Export filtered results to file")
    parser.add_argument("--export-format", choices=["json", "csv", "text"], default="json", help="Export format")
    parser.add_argument("--save-report", help="Save report to file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LogAnalyzer(args.config)
    
    # Parse log files
    analyzer.parse_files(args.files)
    
    if not analyzer.entries:
        print("No log entries found!")
        return
    
    # Apply filters
    start_time = None
    end_time = None
    
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Invalid start time format: {args.start_time}")
            return
    
    if args.end_time:
        try:
            end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Invalid end time format: {args.end_time}")
            return
    
    filtered_entries = analyzer.filter_entries(
        start_time=start_time,
        end_time=end_time,
        levels=args.levels,
        sources=args.sources,
        pattern=args.pattern
    )
    
    print(f"Filtered to {len(filtered_entries):,} entries")
    
    # Perform analysis
    analysis = analyzer.analyze(filtered_entries)
    
    # Generate report
    report = analyzer.generate_report(analysis, args.output_format)
    
    if args.save_report:
        with open(args.save_report, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.save_report}")
    else:
        print(report)
    
    # Export filtered entries if requested
    if args.export:
        analyzer.export_filtered_logs(filtered_entries, args.export, args.export_format)

if __name__ == "__main__":
    main()