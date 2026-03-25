#!/usr/bin/env bash
set -euo pipefail

# Simple helper for: git add -A -> git commit -> git push
# Usage examples:
#   ./push_github.sh
#   ./push_github.sh "Fix reward calculation"
#   ./push_github.sh "Update docs" main origin

COMMIT_MSG="${1:-Update project files}"
BRANCH="${2:-$(git branch --show-current)}"
REMOTE="${3:-origin}"

if [[ -z "$BRANCH" ]]; then
  echo "Cannot detect current branch. Please pass branch name as arg #2."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Current directory is not a git repository."
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Remote '$REMOTE' not found."
  exit 1
fi

echo "==> Repository status before push"
git status -sb

git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "$COMMIT_MSG"
fi

echo "==> Pushing to $REMOTE/$BRANCH"
git push "$REMOTE" "$BRANCH"

echo "==> Final status"
git status -sb
git log --oneline -n 2
