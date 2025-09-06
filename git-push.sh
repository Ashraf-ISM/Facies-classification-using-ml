#!/bin/bash
# Script to push changes to GitHub automatically

# Check repo status
git status

# Stage all changes
git add .

# Commit with a custom message (use default if not given)
commit_message=${1:-"Updated MLP classifier notebook with confusion matrix and accuracy metrics"}
git commit -m "$commit_message"

# Push to main branch
git push origin main
echo "Changes pushed to GitHub successfully."