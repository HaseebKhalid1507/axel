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
# It also fails if anything in the real HOME's axel folders (see `watched`
# below) changed during the run. Another program that writes those folders
# while the gate runs triggers this too.
#
# Before anything is built it fails, with file and line, if a .rs file under
# crates/ calls set_var, remove_var or env_remove on a line that names HOME,
# XDG_ or TMPDIR, or calls env_clear at all.
#
# Never writes to the real HOME: the moved HOME is a fresh temp directory,
# removed on exit. Not caught: a write that is undone before the run ends,
# and writes outside the moved HOME and the watched folders. Needs GNU find
# (Linux).

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Keep the toolchain reachable once HOME moves.
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"

# No Rust source may re-point or remove HOME, an XDG directory or TMPDIR:
# set_var moves them for every thread of a test binary, and a child without
# HOME falls back to the passwd entry, the real home. Either way the write
# lands where the checks below may not look. env_clear() removes HOME
# whatever else its line says, so it is refused on any line. Checked first:
# it needs no build.
src=crates
calls="$(grep -rnE --include='*.rs' '(set_var|remove_var|env_clear|env_remove)' "$src")"
rc=$?
if [ ! -d "$src" ] || [ "$rc" -gt 1 ]; then
    echo "dirty-home: FAIL: could not search $src for environment overrides" >&2
    exit 1
fi
overrides="$(printf '%s\n' "$calls" | grep -E '^[^:]+:[0-9]+:.*(env_clear|HOME|XDG_|TMPDIR)')"
if [ -n "$overrides" ]; then
    echo "dirty-home: FAIL: these lines change HOME, an XDG directory or TMPDIR of the process or of a child (file:line):" >&2
    printf '%s\n' "$overrides" >&2
    echo "dirty-home: pass the folder to the code under test instead; nothing was built or run" >&2
    exit 1
fi

# Build under the real HOME, so the moved HOME sees only test runs.
cargo test --locked --no-run "$@" || exit $?

# The real HOME, and the folders in it that this code base reads or writes.
# Not the whole .synaps-cli: a live Synaps session writes its session files
# there all the time, and the gate would fail for a reason that is not the
# tests.
real="$HOME"
watched=(.config/axel .cache/axel .cache/velocirag .synaps-cli/memory .synaps-cli/plugins/axel)

home="$(mktemp -d "${TMPDIR:-/tmp}/axel-dirty-home.XXXXXX")" || exit 1
# The stamp is outside both homes. Anything in the watched folders that is
# newer than it after the run was written during the run.
stamp="$(mktemp "${TMPDIR:-/tmp}/axel-dirty-home-stamp.XXXXXX")" || exit 1
trap 'rm -rf "$home" "$stamp"' EXIT
# File times can be as coarse as a second: let the stamp age before the run,
# so that a write right at its start is newer.
sleep 1

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

# The real HOME: nothing in the watched folders may be newer than the stamp.
# A test gets there with HOME unset (the passwd entry is used instead) or
# with an absolute path. A folder that does not exist is skipped.
present=()
for folder in "${watched[@]}"; do
    [ -e "$real/$folder" ] && present+=("$real/$folder")
done
if [ ! -d "$real" ] || [ ! -e "$stamp" ]; then
    echo "dirty-home: FAIL: the real HOME ($real) or the stamp file is gone; nothing could be checked" >&2
    [ "$status" -eq 0 ] && status=1
elif [ "${#present[@]}" -eq 0 ]; then
    echo "dirty-home: none of the watched folders exists in the real HOME ($real)"
elif ! touched="$(find -H "${present[@]}" \( -newer "$stamp" -o -cnewer "$stamp" \) -print | LC_ALL=C sort)"; then
    echo "dirty-home: FAIL: could not search the watched folders of the real HOME ($real)" >&2
    [ "$status" -eq 0 ] && status=1
elif [ -n "$touched" ]; then
    echo "dirty-home: FAIL: written in the real HOME during the run ($(printf '%s\n' "$touched" | wc -l) paths):" >&2
    printf '%s\n' "$touched" | head -n 60 >&2
    echo "dirty-home: (another program that writes these folders during the run also triggers this; if so, stop it and run the gate again)" >&2
    [ "$status" -eq 0 ] && status=1
else
    echo "dirty-home: the real HOME is untouched (${#present[@]} of ${#watched[@]} watched folders exist)"
fi

if [ "$status" -ne 0 ]; then
    echo "dirty-home: FAIL (exit $status)" >&2
fi
exit "$status"
