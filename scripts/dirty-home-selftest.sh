#!/usr/bin/env bash
# dirty-home-selftest.sh: proves each failure of scripts/dirty-home.sh.
#
# Runs the gate with a stub in place of cargo, a temp folder in place of the
# real HOME and a temp folder in place of crates/. Compiles nothing, touches
# nothing outside its own temp folder, takes a few seconds.
#
# Usage: bash scripts/dirty-home-selftest.sh     (exit 0: every case behaved)

set -uo pipefail

gate="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dirty-home.sh"
work="$(mktemp -d "${TMPDIR:-/tmp}/axel-dirty-home-selftest.XXXXXX")" || exit 1
trap 'rm -rf "$work"' EXIT

# The stub. The gate calls it twice: `test --locked --no-run` (the build,
# under the real HOME) and `test --locked` (the run, under the moved HOME).
# It acts in the run only, as STUB_MODE says.
cat > "$work/cargo-stub" <<'STUB'
#!/usr/bin/env bash
for arg in "$@"; do [ "$arg" = --no-run ] && exit 0; done
case "$STUB_MODE" in
    nothing) ;;
    moved-home) mkdir -p "$HOME/.cache/axel" && echo x > "$HOME/.cache/axel/written" ;;
    real-home) echo x > "$STUB_REAL/.cache/axel/written" ;;
    real-home-unwatched) echo x > "$STUB_REAL/.synaps-cli/session.json" ;;
    delete-home) rm -rf "$HOME" ;;
    test-fails) exit 101 ;;
    *) echo "cargo-stub: unknown STUB_MODE '$STUB_MODE'" >&2; exit 99 ;;
esac
exit 0
STUB
chmod +x "$work/cargo-stub"

# A `find` that fails, for the case where the gate cannot look.
mkdir "$work/broken-find"
printf '%s\n' '#!/usr/bin/env bash' 'echo "find: broken on purpose" >&2' 'exit 1' > "$work/broken-find/find"
chmod +x "$work/broken-find/find"

failures=0

# case <name> <pass|fail> <text the output must contain> <stub mode> <source line> [folder put first in PATH]
case_() {
    local name="$1" want="$2" text="$3" mode="$4" line="$5" path="${6:+$6:}$PATH"
    local real="$work/$name/real" src="$work/$name/crates"
    # Three of the five watched folders exist, one with an old file in it;
    # the other two are missing and must be skipped.
    mkdir -p "$real/.cache/axel" "$real/.config/axel" "$real/.synaps-cli/memory" "$src/a/src"
    echo old > "$real/.config/axel/config.toml"
    printf '%s\n' 'fn main() {}' "$line" > "$src/a/src/lib.rs"

    local out rc got
    out="$(PATH="$path" CARGO="$work/cargo-stub" DIRTY_HOME_REAL="$real" DIRTY_HOME_SRC="$src" \
        STUB_MODE="$mode" STUB_REAL="$real" bash "$gate" -p stub 2>&1)"
    rc=$?
    if [ "$rc" -eq 0 ]; then got=passed; else got=failed; fi
    if [ "$got" = "${want}ed" ] && printf '%s\n' "$out" | grep -qF -- "$text"; then
        echo "ok    $name: gate $got (exit $rc), says \"$text\""
    else
        echo "WRONG $name: wanted the gate to $want and say \"$text\"; it $got (exit $rc) with:"
        printf '%s\n' "$out" | sed 's/^/      /'
        failures=$((failures + 1))
    fi
}

case_ writes-nothing       pass "the real HOME is untouched (3 of 5"      nothing             ''
case_ unwatched-synaps     pass "the moved HOME is still empty"           real-home-unwatched ''
case_ writes-moved-home    fail "the tests wrote into the moved HOME"     moved-home          ''
case_ writes-real-home     fail "/real/.cache/axel/written"               real-home           ''
case_ real-home-message    fail "another program that writes these"       real-home           ''
case_ deletes-moved-home   fail "no longer exists"                        delete-home         ''
case_ find-is-broken       fail "could not list the moved HOME"           nothing             '' "$work/broken-find"
case_ test-fails           fail "FAIL (exit 101)"                         test-fails          ''
case_ source-set-var       fail "a/src/lib.rs:2:"                         nothing             'fn f() { std::env::set_var("XDG_CACHE_HOME", "/x"); }'
case_ source-env-remove    fail "a/src/lib.rs:2:"                         nothing             'fn f(c: &mut Command) { c.env_remove("HOME"); }'
case_ source-env-clear     fail "a/src/lib.rs:2:"                         nothing             'fn f(c: &mut Command) { c.env_clear(); }'
case_ source-other-var     pass "the moved HOME is still empty"           nothing             'fn f() { std::env::set_var("AXEL_SOURCES", "/x"); }'

if [ "$failures" -ne 0 ]; then
    echo "dirty-home-selftest: FAIL ($failures cases)"
    exit 1
fi
echo "dirty-home-selftest: all cases behaved"
