#!/usr/bin/env bash
set -euo pipefail

readonly root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

required_files=(
  "AGENTS.md"
  "PROJECT_WORKLOG.md"
  "SECURITY.md"
  "README.md"
  "docs/RESEARCH_DESIGN.md"
)

for file in "${required_files[@]}"; do
  test -s "$file" || { echo "missing or empty required file: $file" >&2; exit 1; }
done

rg --fixed-strings --quiet "# 1. Project goal and non-goals" PROJECT_WORKLOG.md
rg --fixed-strings --quiet "# 6. Current next step" PROJECT_WORKLOG.md
rg --fixed-strings --quiet "## Threat model" SECURITY.md
rg --fixed-strings --quiet "## Explicitly unsupported guarantees" SECURITY.md
rg --fixed-strings --quiet "## 1. Research question" docs/RESEARCH_DESIGN.md
rg --fixed-strings --quiet "## 6. Research stages" docs/RESEARCH_DESIGN.md

next_step_count="$(rg --count '^\*\*唯一下一步：' PROJECT_WORKLOG.md)"
test "$next_step_count" = "1" || { echo "PROJECT_WORKLOG.md must contain exactly one global next step" >&2; exit 1; }

compute_status_count="$(rg --count '^\*\*计算资源：`(LOCAL_OK|SERVER_REQUIRED)`' PROJECT_WORKLOG.md)"
test "$compute_status_count" = "1" || { echo "PROJECT_WORKLOG.md must contain exactly one compute-resource status" >&2; exit 1; }

echo "governance documents passed"
