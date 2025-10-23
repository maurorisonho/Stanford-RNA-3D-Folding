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


# Git History Analyzer for Stanford RNA 3D Folding
# Simple script to check and clean Git history

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Git History Analysis - Stanford RNA 3D Folding${NC}"
echo "=================================================="
echo ""

# Check if in git repo
if [ ! -d .git ]; then
    echo -e "${RED}[ERROR] Not a Git repository${NC}"
    exit 1
fi

echo -e "${BLUE}[RESULT] Repository Statistics:${NC}"
TOTAL_COMMITS=$(git rev-list --count HEAD)
CURRENT_BRANCH=$(git branch --show-current)
echo "  Total commits: $TOTAL_COMMITS"
echo "  Current branch: $CURRENT_BRANCH"
echo ""

echo -e "${BLUE}[CHECKING] Analyzing commit messages for common hygiene issues:${NC}"
echo ""

SHORT_COUNT=0
WIP_COUNT=0
FIXUP_COUNT=0
PROBLEMATIC_COMMITS=0

TEMP_FILE=$(mktemp)
git log --pretty=format:"%h %s" > "$TEMP_FILE"

while IFS= read -r line; do
    HASH=$(echo "$line" | cut -d' ' -f1)
    MESSAGE=$(echo "$line" | cut -d' ' -f2-)

    HAS_ISSUE=false

    if [ "${#MESSAGE}" -lt 8 ]; then
        SHORT_COUNT=$((SHORT_COUNT + 1))
        HAS_ISSUE=true
        echo "  [WARNING]  $HASH: message shorter than 8 characters"
    fi

    if echo "$MESSAGE" | grep -qiE "\bWIP\b"; then
        WIP_COUNT=$((WIP_COUNT + 1))
        HAS_ISSUE=true
        echo "  [WARNING]  $HASH: contains 'WIP'"
    fi

    if echo "$MESSAGE" | grep -qi "^fixup!"; then
        FIXUP_COUNT=$((FIXUP_COUNT + 1))
        HAS_ISSUE=true
        echo "  [WARNING]  $HASH: fixup commit"
    fi

    if [ "$HAS_ISSUE" = true ]; then
        PROBLEMATIC_COMMITS=$((PROBLEMATIC_COMMITS + 1))
        echo "      └── $MESSAGE"
    fi
done < "$TEMP_FILE"

rm -f "$TEMP_FILE"

echo ""
echo -e "${BLUE}[ANALYSIS] Analysis Results:${NC}"
echo "  Short messages: $SHORT_COUNT"
echo "  Work-in-progress markers: $WIP_COUNT"
echo "  Fixup commits: $FIXUP_COUNT"
echo "  Total commits flagged: $PROBLEMATIC_COMMITS"
echo ""

if [ $PROBLEMATIC_COMMITS -eq 0 ]; then
    echo -e "${GREEN}[OK] Excellent! Your repository history is clean.${NC}"
    echo -e "${GREEN}   No hygiene issues detected in commit messages.${NC}"
    echo ""
    echo -e "${BLUE}[SUCCESS] No action needed - your Git history is already professional!${NC}"
else
    echo -e "${YELLOW}[WARNING]  Found $PROBLEMATIC_COMMITS commits that contain problematic patterns.${NC}"
    echo ""
    echo -e "${BLUE}[TIP] Recommended actions:${NC}"
    echo "1. For future commits, avoid extremely short messages"
    echo "2. Remove work-in-progress markers before pushing"
    echo "3. Squash fixup commits into their targets when possible"
    echo ""
    echo -e "${YELLOW}[INFO] To clean history (if needed):${NC}"
    echo "1. Create backup: git branch backup-original"
    echo "2. Use git filter-branch or git filter-repo to clean messages"
    echo "3. Force push to remote ([WARNING]  destructive operation)"
fi

echo ""
echo -e "${BLUE}[PROCESS] Recent commit history:${NC}"
git log --oneline -10

echo ""
echo -e "${BLUE}Analysis complete.${NC}"
