#!/usr/bin/env bash
# One-time GitHub setup: branch protection + auto-merge.
# This is your last line of defence if the agent misbehaves.
# Run it once and forget it.
#
# Two ways to run it — both need ADMIN permission on the repo:
#   In the container (no host gh login needed):
#     security add-generic-password -s gh-admin -a "$USER" -w '<admin PAT>'
#     ./run.sh --github-setup
#   On the host, with your own gh session:
#     ./setup-github.sh

set -euo pipefail

command -v gh >/dev/null || { echo "error: gh CLI not found"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: no gh credentials (login with 'gh auth login' or provide GH_TOKEN)"; exit 1; }

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
CHECK="${CHECK:-}"       # empty = auto-detect from the checks that ran on main

# ── Resolve the required status check name ───────────────────────────────────
# Branch protection that requires a check which never reports = nothing ever
# merges, silently. So instead of guessing, ask GitHub which checks actually
# ran on main and validate/auto-detect against that list.
#
# Source: Actions job names via the workflow-runs API (a job's name IS its
# status-check context). Not the check-runs API — fine-grained PATs have no
# "Checks" permission, but "Actions: Read-only" covers this.
detect_checks() {
  gh api "repos/$REPO/actions/runs?branch=main&per_page=10" \
      --jq '.workflow_runs[].id' 2>/dev/null \
    | head -5 \
    | while read -r run_id; do
        gh api "repos/$REPO/actions/runs/$run_id/jobs?per_page=100" \
          --jq '.jobs[].name' 2>/dev/null || true
      done | sort -u
}
NAMES="$(detect_checks || true)"

# CHECK accepts a comma-separated list: CHECK="Lint & Format,Tests (3.12)".
# IMPORTANT: only require checks that run on EVERY pull request. A job that
# only runs on main, tags or releases (deploy, release-please, publish...)
# will never report on a PR — requiring it means nothing can ever merge.
CHECK_LIST=()
if [[ -n "$CHECK" ]]; then
  IFS=',' read -r -a CHECK_LIST <<<"$CHECK"
  for i in "${!CHECK_LIST[@]}"; do
    CHECK_LIST[$i]="$(echo "${CHECK_LIST[$i]}" | sed 's/^ *//; s/ *$//')"
  done
  for c in "${CHECK_LIST[@]}"; do
    if [[ -n "$NAMES" ]] && ! grep -qxF "$c" <<<"$NAMES"; then
      echo "error: no check named '$c' has reported on main." >&2
      echo "Checks seen on main:" >&2
      sed 's/^/  - /' <<<"$NAMES" >&2
      echo "Re-run with CHECK set to one or more of the above (comma-separated)." >&2
      exit 1
    fi
  done
elif [[ -n "$NAMES" ]] && grep -qxF "ci" <<<"$NAMES"; then
  CHECK_LIST=(ci)
elif [[ -n "$NAMES" && "$(wc -l <<<"$NAMES" | tr -d ' ')" == "1" ]]; then
  CHECK_LIST=("$NAMES")
  echo "note: auto-detected required check '$NAMES' (the only check reporting on main)"
elif [[ -z "$NAMES" ]]; then
  CHECK_LIST=(ci)
  echo "note: no check runs found on main yet — defaulting to 'ci'."
  echo "      If your CI job is named differently, re-run with CHECK=<name>."
else
  echo "error: several checks report on main and none is named 'ci':" >&2
  sed 's/^/  - /' <<<"$NAMES" >&2
  echo "Pick the one(s) that must gate merges — comma-separated, and ONLY jobs" >&2
  echo "that run on every pull request (never release/deploy-only jobs):" >&2
  echo "  CHECK=\"Lint & Format,Tests (3.12)\" ./run.sh --github-setup" >&2
  exit 1
fi

CONTEXTS_JSON="$(printf '"%s",' "${CHECK_LIST[@]}")"
CONTEXTS_JSON="[${CONTEXTS_JSON%,}]"

echo "repo:            $REPO"
echo "required checks: ${CHECK_LIST[*]}"
echo
if [[ "${AGENTLOOP_YES:-}" == "1" ]]; then
  echo "AGENTLOOP_YES=1 — proceeding without confirmation."
else
  read -rp "This will lock down main and enable auto-merge. Continue? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted."; exit 0; }
fi

echo
echo "> branch protection (main)"
# Tolerant: on private repos under the Free plan GitHub rejects branch
# protection (403). Warn and continue — labels and the rest still matter.
set +e
gh api -X PUT "repos/$REPO/branches/main/protection" \
  --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": $CONTEXTS_JSON
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
echo "  OK  main protected: force push off, CI required"

echo
echo "> repository settings"
PROT_RC=$?
if [[ $PROT_RC -ne 0 ]]; then
  echo "  WARN branch protection failed (private repo on the Free plan?)."
  echo "       Merges will not be gated by required checks; the automerge"
  echo "       workflow falls back to watching CI itself. Continuing."
fi
gh repo edit --enable-auto-merge --enable-squash-merge --delete-branch-on-merge \
  || echo "  WARN could not enable repo auto-merge (plan limitation) — the automerge workflow's fallback covers this"
set -e
echo "  OK  squash + delete branch on merge (auto-merge: see above)"

echo
echo "> labels"
gh label create automated --color 0e8a16 --description "Opened by the agent, auto-merge when green" --force >/dev/null
gh label create blocked   --color d93f0b --description "Agent is stuck, needs a human" --force >/dev/null
gh label create plan      --color 1d76db --description "Change to PLAN.md, review before merging" --force >/dev/null
echo "  OK  automated, blocked, plan"

cat <<'EOF'

========================================================
Done. Two manual steps remain:

1) Give the agent token access to this repo.
   ONE token serves all your agentloop projects — if you already have it,
   just add this repo to its list:
     github.com/settings/personal-access-tokens
       -> your agent token -> Repository access -> add this repo
   Creating it for the first time:
     github.com/settings/personal-access-tokens/new
     · Repository access -> Only select repositories -> your agentloop repos
     · Permissions -> Contents: Read and write
     · Permissions -> Pull requests: Read and write
     · Permissions -> Actions: Read-only   (CI verdict polling — note:
       GitHub offers no "Checks" permission on fine-grained PATs)
     · Select NOTHING else (no admin, no org scopes)

   Then store it in the keychain:
     security add-generic-password -s gh-agent -a "$USER" -w 'github_pat_...'

2) Let Actions open and merge pull requests:
   Settings -> Actions -> General -> Workflow permissions
     · Select "Read and write permissions"
     · Check "Allow GitHub Actions to create and approve pull requests"
========================================================
EOF
