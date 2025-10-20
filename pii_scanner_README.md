# PII Scanner - Automated Personal Information Detector

This Python script automates the detection of **PII (Personally Identifiable Information)** in code repositories, enabling reproducible and systematic scanning for sensitive data.

## Features

### PII Types Detected
- **Emails** (medium priority)
- **Private keys** SSH/RSA (critical)
- **AWS keys** (critical/high)
- **Credentials** (passwords, tokens, API keys)
- **Brazilian CPF** (high)
- **US SSN** (high)
- **Brazilian phone numbers** (medium)
- **Credit cards** (critical)

### Smart Filters
- Automatically ignores `.venv`, `node_modules`, `.git`
- Detects false positives (scientific coordinates, examples)
- Filters binary and oversized files

## How to Use

### Installation
```bash
# The script is standalone, just ensure you have Python 3.8+
chmod +x pii_scanner.py
```

### Basic Usage
```bash
# Scan current directory
./pii_scanner.py

# Scan specific directory
./pii_scanner.py /path/to/project

# Generate JSON report
./pii_scanner.py --format json

# Save report to file
./pii_scanner.py --output pii_report.txt
```

### Advanced Options
```bash
# Exclude custom patterns
./pii_scanner.py --exclude ".*\.cif$" --exclude "test_.*"

# No colors (for CI/CD)
./pii_scanner.py --no-color

# Complete example
./pii_scanner.py stanford_rna3d/ \
  --format json \
  --output pii_report.json \
  --exclude ".*\.pdb$" \
  --exclude "data/raw/.*"
```

##  Results Interpretation

### Priority Levels
- **CRITICAL**: Private keys, credit cards - **IMMEDIATE ACTION**
- **HIGH**: CPF, SSN, credentials - **Review urgently**
- **MEDIUM**: Emails, phones - **Verify if appropriate**
- **LOW**: Possibly public data - **Document**

### Exit Codes
- `0`: No issues or only low priority findings
- `1`: Critical or high priority findings detected

## Git Hooks Integration

### Pre-commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python3 pii_scanner.py --no-color
exit_code=$?

if [ $exit_code -eq 1 ]; then
    echo "ERROR: Possible PII detected! Commit blocked."
    echo "Run 'python3 pii_scanner.py' for details."
    exit 1
fi
```

### GitHub Actions
```yaml
name: PII Scanner
on: [push, pull_request]
jobs:
  pii-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run PII Scanner
        run: |
          python3 pii_scanner.py --format json --output pii_report.json
      - name: Upload results
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: pii-report
          path: pii_report.json
```

## Use Cases

### 1. Security Audit
```bash
# Complete scan with detailed report
./pii_scanner.py . --format json --output audit_$(date +%Y%m%d).json
```

### 2. Pre-Release Verification
```bash
# Only critical/high issues
./pii_scanner.py src/ && echo "Ready for release"
```

### 3. Repository Cleanup
```bash
# Find all PII types
./pii_scanner.py --exclude "\.git.*" > cleanup_report.txt
```

## Customization

### Add New Patterns
Edit the `PATTERNS` constant in the script:
```python
'custom_pattern': {
    'regex': r'your_regex_here',
    'severity': 'high',
    'description': 'Pattern description'
}
```

### Custom Filters
Modify `_is_false_positive()` for project-specific cases.

## Output Example

```
============================================================
PII SCAN REPORT
============================================================
Date: 2024-12-19 10:30:15
Directory: /home/user/project
Total findings: 5

HIGH PRIORITY (2 findings)
--------------------------------------------------
File: config/settings.py:15
Type: secret_keywords
Text: api_key = "sk-abc123..."
Context: OPENAI_API_KEY = "sk-abc123..."

MEDIUM PRIORITY (3 findings)
--------------------------------------------------
File: docs/README.md:42
Type: email
Text: contact@company.com
Context: Contact us: contact@company.com
```

## Performance

- **Speed**: ~100 files/second
- **Memory**: Low consumption, processes line by line
- **Scalability**: Suitable for repositories with thousands of files

## Limitations and Considerations

### What is NOT detected:
- PII in binary data
- Obfuscated or encrypted data
- PII in images or PDFs
- Region-specific patterns from other countries

### Common False Positives:
- Scientific coordinates resembling CPF
- Emails in documentation
- Test/example keys
- IDs that look like credentials

## Maintenance

Update regularly:
1. **Regex patterns** for new PII types
2. **False positive filters** based on experience
3. **Exclusion lists** as the project evolves

---

**Tip**: Run the scanner regularly during development to detect PII before it reaches the remote repository!