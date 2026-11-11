#!/usr/bin/env bash
# dirty-home.sh: the test gate.
#
# Builds the test binaries under the real HOME, then runs the suite with HOME
# and the XDG base directories pointing into an empty temp directory, and
# fails if anything exists in that directory afterwards.
#
# A test that reads or writes the developer's home (a brain, a model cache,
# a config file) passes on a laptop and is still wrong. Here it fails: what
# it wanted to read is not there, and what it wrote is found.
#
# The build step keeps the real HOME because ort's build script looks for
# its downloaded ONNX Runtime library in the user's cache directory.
#
# Usage: bash scripts/dirty-home.sh [cargo test args]
#   The gate: bash scripts/dirty-home.sh -p axel -p axel-memkoshi -p velocirag
#   --locked is always passed; do not repeat it.
#
# Never writes to the real HOME: the moved HOME is a fresh temp directory,
# removed on exit. Not caught: a write that is undone before the run ends,
# and writes outside HOME. Needs GNU find (Linux).

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep the toolchain reachable once HOME moves.
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"

# Build under the real HOME, so the moved HOME sees only test runs.
cargo test --locked --no-run "$@" || exit $?

home="$(mktemp -d "${TMPDIR:-/tmp}/axel-dirty-home.XXXXXX")" || exit 1
trap 'rm -rf "$home"' EXIT

# None of the XDG directories is created: the directory starts empty, and a
# test that creates one of them is a test that wrote into HOME.
# A failing test must not skip the check below: run it either way, then
# report cargo's status.
status=0
env -u AXEL_BRAIN \
    HOME="$home" \
    XDG_CACHE_HOME="$home/.cache" \
    XDG_CONFIG_HOME="$home/.config" \
    XDG_DATA_HOME="$home/.local/share" \
    XDG_STATE_HOME="$home/.local/state" \
    XDG_RUNTIME_DIR="$home/.runtime" \
    cargo test --locked "$@" || status=$?

# The gate must fail when it could not look: a moved HOME that is gone, or a
# `find` that failed, is not "still empty".
looked=1
left=""
if [ ! -d "$home" ]; then
    echo "dirty-home: FAIL: the moved HOME ($home) no longer exists; the run removed it, so nothing could be checked" >&2
    looked=0
elif ! left="$(find "$home" -mindepth 1 -printf '%y %s %P\n' | LC_ALL=C sort)"; then
    echo "dirty-home: FAIL: could not list the moved HOME ($home); GNU find is needed" >&2
    looked=0
fi
if [ "$looked" -eq 0 ]; then
    [ "$status" -eq 0 ] && status=1
elif [ -n "$left" ]; then
    echo "dirty-home: FAIL: the tests wrote into the moved HOME ($(printf '%s\n' "$left" | wc -l) entries; type, bytes, path):" >&2
    printf '%s\n' "$left" | head -n 60 >&2
    [ "$status" -eq 0 ] && status=1
else
    echo "dirty-home: the moved HOME is still empty"
fi
if [ "$status" -ne 0 ]; then
    echo "dirty-home: FAIL (exit $status)" >&2
fi
exit "$status"
