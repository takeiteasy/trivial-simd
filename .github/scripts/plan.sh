#!/usr/bin/env bash
# Chooses which CI jobs run and writes matrix, extras and count to GITHUB_OUTPUT.
# Env: EVENT_NAME REF ACTION LABEL PR_LABELS BASE_SHA HEAD_SHA INPUT_OS INPUT_LISP
set -euo pipefail

config=${CI_CONFIG:-.github/ci-matrix.json}
output=${GITHUB_OUTPUT:-/dev/stdout}
os=${INPUT_OS:-all}
lisp=${INPUT_LISP:-}
zero=0000000000000000000000000000000000000000

message=$(git log -1 --format=%B "$HEAD_SHA")
full=false
[[ $EVENT_NAME == workflow_dispatch ]] && full=true
[[ $EVENT_NAME == push && ${REF:-} == refs/tags/* ]] && full=true
[[ $message == *'[ci full]'* ]] && full=true
[[ ",${PR_LABELS:-}," == *,ci-full,* ]] && full=true

skip=false
[[ ${ACTION:-} == labeled && ${LABEL:-} != ci-full ]] && skip=true

if [[ -z ${BASE_SHA:-} || $BASE_SHA == "$zero" ]] || ! git cat-file -e "$BASE_SHA^{commit}" 2>/dev/null; then
  changed=$(git ls-tree -r --name-only "$HEAD_SHA")
else
  changed=$(git diff --name-only "$BASE_SHA...$HEAD_SHA")
fi

docs_re=$(jq -r '.paths.docs' "$config")
focus_re=$(jq -r '.paths.focus // "a^"' "$config")
docs_only=false
if [[ -n $changed ]] && ! grep -qvE "$docs_re" <<<"$changed"; then docs_only=true; fi
focus=false
if grep -qE "$focus_re" <<<"$changed"; then focus=true; fi

selected=$(jq -c \
  --argjson full "$full" --argjson skip "$skip" --argjson docs "$docs_only" \
  --argjson focus "$focus" --arg os "$os" --arg lisp "$lisp" '
  (.entries + .extras) | map(select(
    if $skip then false
    elif $full then
      (($os == "all") or (.family | index($os))) and (($lisp == "") or (.lisp | contains($lisp)))
    else
      .tier == 1 and ($docs | not) and (.when == "any" or (.when == "focus" and $focus))
    end))' "$config")

matrix=$(jq -c 'map(select(has("job") | not))' <<<"$selected")
extras=$(jq -c 'map(select(has("job")) | .job)' <<<"$selected")
count=$(jq 'length' <<<"$matrix")

{
  echo "matrix=$matrix"
  echo "extras=$extras"
  echo "count=$count"
} >>"$output"

{
  echo "full=$full docs_only=$docs_only focus_changed=$focus skip=$skip"
  echo "jobs: $(jq -r 'map(.name) | join(", ")' <<<"$selected")"
} | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}" >&2
