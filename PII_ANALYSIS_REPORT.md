# PII Analysis Report (Personally Identifiable Information)

**Analysis Date**: October 19, 2025  
**Repository**: Stanford-RNA-3D-Folding  
**Scope**: Comprehensive analysis excluding `.venv` directories  

## Executive Summary

This report presents the results of an automated scan for personally identifiable information (PII) in the repository. The analysis focused on identifying emails, private keys, tokens, CPFs, phone numbers, and other sensitive data.

### Key Findings

- **200+ emails found**: Primarily in PDB_RNA/*.cif files (public PDB data)
- **0 real private keys** in project code
- **0 AWS credentials/tokens** in main code
- **Author email** present in documentation and notebooks (intentional)
- **Suspicious numbers** only in scientific CSV datasets (coordinates/IDs)

## Findings Classification by Priority

### HIGH PRIORITY (Immediate Action Recommended)

#### 1. Virtual Environment Directories
**Location**: `.venv/` and `stanford_rna3d/.venv/`
**Issue**: These directories contain third-party packages and assets that should not be in the repository
**Action**: Remove from repository and add to .gitignore

### MEDIUM PRIORITY (Review Recommended)

#### 2. Emails in PDB Files
**Location**: `stanford_rna3d/data/raw/PDB_RNA/*.cif`
**Examples found**:
- jfh21@columbia.edu
- bwiedenheft@gmail.com  
- wahc@stanford.edu
- rhiju@stanford.edu
- anna.pyle@yale.edu
- alexey.amunts@gmail.com

**Context**: These files contain public metadata from the Protein Data Bank (PDB) including contact information of experiment authors.
**Risk**: Low - data already public in PDB
**Recommendation**: Keep + document origin, or redact if privacy policy requires

#### 3. Project Author Email
**Location**: Documentation, notebooks, LICENSE
**Email**: mauro.risonho@gmail.com
**Context**: Authorship attribution intentionally included
**Risk**: Low - voluntary PII for attribution
**Recommendation**: Keep if public authorship is desired

### LOW PRIORITY (Informational)

#### 4. RNA Sequence Data
**Location**: `stanford_rna3d/data/raw/train_labels.csv`, `validation_labels.csv`
**Content**: Sequence IDs (e.g., R1128_37), numerical coordinates
**Risk**: Very low - scientific data, not personal PII

## Technical Details

### Patterns Searched
- Emails: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
- Private keys: `-----BEGIN.*PRIVATE KEY-----`
- AWS Access Keys: `AKIA[0-9A-Z]{16}`, etc.
- Tokens/passwords: `password|secret|api_key|token|credential`
- CPF: `\d{3}\.\d{3}\.\d{3}-\d{2}`
- Phone numbers: Brazilian and international patterns

### Identified False Positives
- Numbers in CSV datasets: X,Y,Z coordinates and sequence IDs
- Assets in .venv: minified JS, package METADATA
- Test data in site-packages

## Recommended Actions

### Action 1: Environment Cleanup - EXECUTE NOW
```bash
# Add to .gitignore
echo ".venv/" >> .gitignore
echo "stanford_rna3d/.venv/" >> .gitignore

# Remove from repository (if versioned)
git rm -r --cached .venv/ stanford_rna3d/.venv/ 2>/dev/null || true
git add .gitignore
git commit -m "Remove virtual environments from repository and update .gitignore"
```

### Action 2: PDB Data Documentation
- Add note in README about origin of PDB_RNA/*.cif files
- Explain that they contain public PDB metadata including author contacts

### Action 3: Monitoring Script (Optional)
- Create Python script for automated PII scanning
- Configure to run before commits

## No Critical Credentials Found

The analysis **did not find**:
- Real SSH/TLS private keys in code
- Valid API tokens
- Plaintext passwords
- Real AWS credentials
- Sensitive personal data (CPF, credit cards) in code

## Conclusion

The repository is in **good security condition**. The main actions needed are:
1. Remove `.venv` directories (housekeeping)
2. Document origin of PDB data (transparency)

No critical security action is immediately necessary.

---
**Automatically generated report** | **Review periodically before public releases**