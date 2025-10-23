#!/bin/bash

# Author: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
# Created: October 18, 2025 at 14:30:00
# License: MIT License
# Kaggle Competition: https://www.kaggle.com/competitions/stanford-rna-3d-folding
#
# ---
#
# MIT License
#
# Copyright (c) 2025 Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---

"""
Git History Cleaner - Standardize commit messages
=================================================

This script cleans the Git history by removing or sanitizing commit messages
that include work-in-progress markers, fixup prefixes, or other inconsistent
patterns from the repository history.

[WARNING]  WARNING: This script rewrites Git history and is DESTRUCTIVE!
Make sure you have backups before running this script.

Usage: bash clean_git_history.sh
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[TOOLS] Git History Cleaner - Stanford RNA 3D Folding${NC}"
echo "=================================================================="
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo -e "${RED}[ERROR] Error: Not in a Git repository root directory${NC}"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}[LOCATION] Current branch: ${CURRENT_BRANCH}${NC}"

# Show current status
echo -e "${BLUE}[RESULT] Current repository status:${NC}"
git log --oneline -5
echo ""

# Ask for confirmation
echo -e "${YELLOW}[WARNING]  WARNING: This operation will rewrite Git history!${NC}"
echo -e "${YELLOW}   This is IRREVERSIBLE and will affect all branches.${NC}"
echo ""
read -p "Do you want to continue? (type 'YES' to confirm): " confirmation

if [ "$confirmation" != "YES" ]; then
    echo -e "${YELLOW}[ERROR] Operation cancelled by user.${NC}"
    exit 0
fi

# Create backup branch
echo -e "${BLUE}💾 Creating backup branch...${NC}"
git checkout -b backup-before-history-clean-$(date +%Y%m%d-%H%M%S)
git checkout $CURRENT_BRANCH

# Function to clean commit message
clean_commit_message() {
    local message="$1"
    
    # Remove leading and trailing whitespace
    cleaned=$(echo "$message" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    
    # Remove common work-in-progress markers
    cleaned=$(echo "$cleaned" | sed -E 's/^[Ww][Ii][Pp][:! -]*//')
    
    # Remove fixup prefixes
    cleaned=$(echo "$cleaned" | sed -E 's/^fixup![[:space:]]*//I')
    
    # Remove excessive whitespace
    cleaned=$(echo "$cleaned" | sed 's/  */ /g' | sed 's/^ *//; s/ *$//')
    
    # If message becomes empty or too short, provide a default
    if [ ${#cleaned} -lt 10 ]; then
        cleaned="Update project files"
    fi
    
    echo "$cleaned"
}

# Export the function so it can be used by git filter-branch
export -f clean_commit_message

echo -e "${BLUE}[PROCESS] Rewriting commit history...${NC}"
echo "This may take a few minutes depending on repository size."
echo ""

# Use git filter-branch to rewrite history
git filter-branch -f --msg-filter '
    message=$(cat)
    clean_commit_message "$message"
' --tag-name-filter cat -- --branches --tags

echo ""
echo -e "${GREEN}[OK] History rewrite completed!${NC}"

# Show the cleaned history
echo ""
echo -e "${BLUE}[RESULT] New commit history (first 10 commits):${NC}"
git log --oneline -10

echo ""
echo -e "${BLUE}🧹 Cleaning up backup files...${NC}"
rm -rf .git/refs/original/

# Force garbage collection
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo ""
echo -e "${YELLOW}[INFO] Next steps:${NC}"
echo "1. Review the cleaned history above"
echo "2. Test your repository to ensure everything works"
echo "3. Force push to remote (if needed):"
echo "   git push --force-with-lease origin main"
echo ""
echo -e "${YELLOW}[WARNING]  Remember: All collaborators will need to clone the repository again!${NC}"

echo ""
echo -e "${GREEN}[SUCCESS] Git history cleaning completed successfully!${NC}"
