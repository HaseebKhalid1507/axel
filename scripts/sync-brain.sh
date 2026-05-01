#!/bin/bash
# Incrementally sync all 6 jawz-search sources into the Axel brain.
# Re-indexes only files whose mtime is newer than indexed_at, and prunes
# DB entries whose file_path no longer exists.
#
# Use this for routine refreshes. Use index-all-sources.sh for cold rebuilds.

set -e

AXEL=~/Projects/axel/target/release/axel
BRAIN=${AXEL_BRAIN:-~/.config/axel/axel.r8}

run_sync() {
    local label="$1" path="$2" source="$3"
    if [ ! -d "$path" ]; then
        echo "→ $label  (skip, missing: $path)"
        return
    fi
    echo "→ $label"
    "$AXEL" --brain "$BRAIN" index-sync "$path" --source "$source"
}

run_sync "mikoshi"         ~/Jawz/mikoshi/Notes/                         mikoshi
run_sync "context"         ~/Jawz/data/context/                          context
run_sync "notes"           ~/Jawz/notes/                                 notes
run_sync "slack-diary"     ~/Jawz/slack/diary/                           slack-diary
run_sync "memories-legacy" ~/Jawz/data/context/memories/permanent/       memories-legacy
run_sync "memories"        ~/.stelline/memkoshi/exports/                 memories

echo "✓ sync done"
