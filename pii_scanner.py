#!/usr/bin/env python3
"""
Automated PII (Personally Identifiable Information) Scanning Script
Stanford RNA 3D Folding Project

This script performs automated scanning for sensitive data in the repository.
Can be used manually or integrated into pre-commit hooks.

Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
License: MIT
"""

import re
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class PIIFinding:
    """Represents a PII finding."""
    file_path: str
    line_number: int
    pattern_type: str
    matched_text: str
    context_line: str
    severity: str


class PIIScanner:
    """Main scanner for PII identification."""

    # Search patterns by type
    PATTERNS = {
        'email': {
            'regex': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            'severity': 'medium',
            'description': 'Email address'
        },
        'private_key': {
            'regex': r'-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+|DSA\s+)?PRIVATE\s+KEY-----',
            'severity': 'critical',
            'description': 'Private key'
        },
        'aws_access_key': {
            'regex': r'\b(AKIA[0-9A-Z]{16}|A3T[A-Z0-9]{16}|ASIA[0-9A-Z]{16}|ACCA[0-9A-Z]{16})\b',
            'severity': 'critical',
            'description': 'AWS access key'
        },
        'aws_secret_key': {
            'regex': r'\b[A-Za-z0-9/+=]{40}\b',
            'severity': 'high',
            'description': 'Potential AWS secret key'
        },
        'secret_keywords': {
            'regex': r'(?i)(password|passwd|secret|api[_-]?key|token|credential)\s*[:=]\s*["\']?[^"\'\s]{8,}["\']?',
            'severity': 'high',
            'description': 'Credential keyword'
        },
        'cpf': {
            'regex': r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',
            'severity': 'high',
            'description': 'Brazilian CPF'
        },
        'ssn': {
            'regex': r'\b\d{3}-\d{2}-\d{4}\b',
            'severity': 'high',
            'description': 'US SSN'
        },
        'phone_br': {
            'regex': r'(\+55\s?)?\(?\d{2}\)?\s?9?\d{4}[- ]?\d{4}',
            'severity': 'medium',
            'description': 'Brazilian phone number'
        },
        'credit_card': {
            'regex': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            'severity': 'critical',
            'description': 'Credit card number'
        }
    }

    # Directories and files to ignore
    IGNORE_PATTERNS = {
        'directories': {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', 'env', '.conda', 'conda-meta',
            '.ipynb_checkpoints', 'build', 'dist', '.egg-info'
        },
        'files': {
            '.gitignore', '.pyc', '.pyo', '.pyd', '.so', '.dylib',
            '.DS_Store', 'Thumbs.db', '.log'
        },
        'extensions': {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.pdf', '.zip',
            '.tar', '.gz', '.rar', '.7z'
        }
    }

    def __init__(self, root_path: str, exclude_patterns: List[str] | None = None,
                 show_progress: bool = True):
        """
        Initialize the scanner.

        Args:
            root_path: Root path for scanning
            exclude_patterns: Additional patterns to exclude
            show_progress: Show progress bar (requires tqdm)
        """
        self.root_path = Path(root_path)
        self.exclude_patterns = set(exclude_patterns or [])
        self.findings: List[PIIFinding] = []
        self.show_progress = show_progress and HAS_TQDM

    def should_ignore_path(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        # Check directories
        for part in path.parts:
            if part in self.IGNORE_PATTERNS['directories']:
                return True

        # Check file name
        if path.name in self.IGNORE_PATTERNS['files']:
            return True

        # Check extension
        if path.suffix.lower() in self.IGNORE_PATTERNS['extensions']:
            return True

        # Check custom patterns
        for pattern in self.exclude_patterns:
            if re.search(pattern, str(path)):
                return True

        return False

    def scan_file(self, file_path: Path) -> List[PIIFinding]:
        """Scan a file for PII."""
        findings: List[PIIFinding] = []

        try:
            # Try reading as UTF-8 text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError, IsADirectoryError):
            return findings

        for line_num, line in enumerate(lines, 1):
            line_content = line.strip()

            # Skip very long lines (likely binary data)
            if len(line_content) > 1000:
                continue

            for pattern_name, pattern_info in self.PATTERNS.items():
                matches = re.finditer(pattern_info['regex'], line_content)

                for match in matches:
                    # Apply specific filters
                    if self._is_false_positive(pattern_name, match.group(), file_path, line_content):
                        continue

                    finding = PIIFinding(
                        file_path=str(file_path.relative_to(self.root_path)),
                        line_number=line_num,
                        pattern_type=pattern_name,
                        matched_text=match.group(),
                        context_line=line_content,
                        severity=pattern_info['severity']
                    )
                    findings.append(finding)

        return findings

    def _is_false_positive(self, pattern_type: str, matched_text: str,
                          file_path: Path, context: str) -> bool:
        """Identify false positives based on context."""

        # Filters for emails
        if pattern_type == 'email':
            # Ignore emails in example comments or documentation
            if any(word in context.lower() for word in
                   ['example', 'exemplo', 'test', 'dummy', 'placeholder']):
                return True

        # Filters for numbers that might be coordinates
        if pattern_type in ['cpf', 'ssn', 'credit_card']:
            # If in CSV file with scientific coordinates
            if file_path.suffix == '.csv' and any(word in context.lower() for word in
                                                 ['coordinate', 'coord', 'xyz', 'position']):
                return True

        # Filters for suspicious AWS keys
        if pattern_type == 'aws_secret_key':
            # If it's a very common string or in a comment
            if matched_text in ['0000000000000000000000000000000000000000',
                               'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'] or \
               context.strip().startswith('#'):
                return True

        return False

    def scan_directory(self) -> List[PIIFinding]:
        """Recursively scan a directory."""
        print(f"Starting scan at: {self.root_path}")

        # Collect all files first for progress bar
        all_files = []
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not self.should_ignore_path(file_path):
                all_files.append(file_path)

        print(f"Found {len(all_files)} files to process")

        # Configure progress bar
        if self.show_progress:
            file_iterator = tqdm(all_files, desc="Scanning files", unit="file")  # type: ignore
        else:
            file_iterator = all_files

        file_count = 0
        for file_path in file_iterator:
            file_count += 1

            # Update progress manually if no tqdm
            if not self.show_progress and file_count % 100 == 0:
                print(f"Processed {file_count}/{len(all_files)} files...")

            file_findings = self.scan_file(file_path)
            self.findings.extend(file_findings)

            # Update bar description with findings
            if self.show_progress and hasattr(file_iterator, 'set_postfix'):
                file_iterator.set_postfix({  # type: ignore
                    'findings': len(self.findings),
                    'file': file_path.name[:20] + ('...' if len(file_path.name) > 20 else '')
                })

        print(f"Scan completed. {file_count} files processed, {len(self.findings)} findings.")
        return self.findings

    def generate_report(self, output_format: str = 'text') -> str:
        """Generate report of findings."""
        if output_format == 'json':
            return self._generate_json_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate report in text format."""
        report = []
        report.append("=" * 60)
        report.append("PII SCAN REPORT")
        report.append("=" * 60)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Directory: {self.root_path}")
        report.append(f"Total findings: {len(self.findings)}")
        report.append("")

        # Group by severity
        by_severity: Dict[str, List[PIIFinding]] = {}
        for finding in self.findings:
            if finding.severity not in by_severity:
                by_severity[finding.severity] = []
            by_severity[finding.severity].append(finding)

        # Ordenar por severidade
        severity_order = ['critical', 'high', 'medium', 'low']
        for severity in severity_order:
            if severity in by_severity:
                findings = by_severity[severity]
                report.append(f"\n{severity.upper()} PRIORITY ({len(findings)} findings)")
                report.append("-" * 50)

                for finding in findings:
                    report.append(f"File: {finding.file_path}:{finding.line_number}")
                    report.append(f"Type: {finding.pattern_type}")
                    report.append(f"Text: {finding.matched_text}")
                    report.append(f"Context: {finding.context_line[:100]}...")
                    report.append("")

        if not self.findings:
            report.append("No PII issues found.")

        return "\n".join(report)

    def _generate_json_report(self) -> str:
        """Generate report in JSON format."""
        data = {
            'scan_info': {
                'timestamp': datetime.now().isoformat(),
                'directory': str(self.root_path),
                'total_findings': len(self.findings)
            },
            'findings': [
                {
                    'file_path': f.file_path,
                    'line_number': f.line_number,
                    'pattern_type': f.pattern_type,
                    'matched_text': f.matched_text,
                    'context_line': f.context_line,
                    'severity': f.severity
                }
                for f in self.findings
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='PII scanner for code repositories'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Directory path to scan (default: current directory)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Report output format'
    )
    parser.add_argument(
        '--output',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--exclude',
        action='append',
        help='Regex patterns to exclude from scan'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable color output'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar'
    )

    args = parser.parse_args()

    # Initialize scanner
    scanner = PIIScanner(args.path, args.exclude, show_progress=not args.no_progress)

    # Warn about tqdm if needed
    if not args.no_progress and not HAS_TQDM:
        print("Tip: Install 'tqdm' for progress bar: pip install tqdm")
        print()

    # Execute scan
    findings = scanner.scan_directory()

    # Generate report
    report = scanner.generate_report(args.format)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    # Exit code
    if findings:
        critical_high = [f for f in findings if f.severity in ['critical', 'high']]
        if critical_high:
            print(f"\nWARNING: {len(critical_high)} critical/high priority findings detected!")
            sys.exit(1)
        else:
            print(f"\nINFO: {len(findings)} low priority findings detected.")
            sys.exit(0)
    else:
        print("\nNo issues found.")
        sys.exit(0)


if __name__ == '__main__':
    main()
