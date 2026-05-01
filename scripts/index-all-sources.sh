#!/bin/bash
# Index all 6 jawz-search sources into the Axel brain.
# Each source is namespaced via --source to prevent doc_id collisions.

set -e

AXEL=~/Projects/axel/target/release/axel
BRAIN=~/.config/axel/axel.r8

echo "→ mikoshi"
$AXEL --brain "$BRAIN" index ~/Jawz/mikoshi/Notes/ --source mikoshi

echo "→ context"
$AXEL --brain "$BRAIN" index ~/Jawz/data/context/ --source context

echo "→ notes"
$AXEL --brain "$BRAIN" index ~/Jawz/notes/ --source notes

echo "→ slack-diary"
$AXEL --brain "$BRAIN" index ~/Jawz/slack/diary/ --source slack-diary

echo "→ memories-legacy"
$AXEL --brain "$BRAIN" index ~/Jawz/data/context/memories/permanent/ --source memories-legacy

echo "→ memories"
$AXEL --brain "$BRAIN" index ~/.stelline/memkoshi/exports/ --source memories

echo "✓ done"
