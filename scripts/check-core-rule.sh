#!/usr/bin/env bash
# check-core-rule.sh <core checkout>
#
# Axel's copy of core's withholding rule (crates/axel/src/core_withheld.txt) must be the
# rule of the given Synaps core checkout. Two comparisons:
#   1. the words: the string literals of the array in `fn credential_like`, sorted, against
#      the file's;
#   2. the whole function: the SHA-256 of its text, from the line with `fn credential_like(`
#      to the first following line that is `}`, bytes as in the file, against the file's
#      `# sha256 <hex>` line. This catches a change of the case rule, of the matching rule,
#      or a second condition, which the words do not show.
#   exit 0  both equal
#   exit 1  different (the difference is printed), or core has no such function or array
#           any more: someone then writes the new mirror
#   exit 2  wrong usage, or a file or a tool is missing
# Only reads. Run by the foreman on a core change that touches the function, and by
# cutover.sh against the checkout the cutover binary was built from.
#
# It does not prove what core does AROUND the function: core applies it to its redactor's
# output and withholds some classes whatever the words (see screen.rs, core_withholds).
#
# RULING: wave2-c8 review A4.

set -uo pipefail
export LC_ALL=C

[ $# -eq 1 ] || { echo "usage: check-core-rule.sh <core checkout>" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
copy="$here/crates/axel/src/core_withheld.txt"
core="$1/crates/agent-core/src/core/context_archive.rs"

[ -f "$copy" ] || { echo "check-core-rule: no $copy" >&2; exit 2; }
[ -f "$core" ] || { echo "check-core-rule: no $core" >&2; exit 2; }

if command -v sha256sum >/dev/null 2>&1; then
    sha() { sha256sum | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha() { shasum -a 256 | cut -d' ' -f1; }
else
    echo "check-core-rule: neither sha256sum nor shasum is installed" >&2; exit 2
fi

# The array: from the first `[` after `fn credential_like(` to the first `]` after it.
theirs="$(awk '
    /fn credential_like\(/ { infn = 1 }
    infn && !inarr && /\[/ { inarr = 1; sub(/^[^[]*\[/, "") }
    inarr {
        line = $0
        closed = sub(/\].*$/, "", line)
        while (match(line, /"([^"\\]|\\.)*"/)) {
            print substr(line, RSTART, RLENGTH)
            line = substr(line, RSTART + RLENGTH)
        }
        if (closed) exit
    }
' "$core" | sort)"

# The function: its lines, each with its line break, hashed.
body() {
    awk '
        !infn && /fn credential_like\(/ { infn = 1 }
        infn { print }
        infn && /^}\r?$/ { exit }
    ' "$core"
}
their_hash=""
if body | grep -q .; then
    their_hash="$(body | sha)"
fi

ours="$(grep -v '^#' "$copy" | grep -v '^[[:space:]]*$' | sort)"
commit="$(head -n 1 "$copy")"
our_hash="$(sed -n 's/^# sha256 \([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$copy" | head -n 1)"

if [ -z "$theirs" ]; then
    echo "check-core-rule: FAIL: no array of string literals in fn credential_like of $core"
    echo "  core's rule has changed shape; the copy ($commit) needs a new mirror"
    exit 1
fi

if [ "$theirs" != "$ours" ]; then
    echo "check-core-rule: FAIL: core_withheld.txt ($commit) differs from fn credential_like in $1"
    echo "  lines with < are only in core, lines with > are only in the copy:"
    diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") | grep '^[<>]' | sed 's/^/  /'
    [ "$their_hash" = "$our_hash" ] || echo "  the function's sha256 differs too: core $their_hash, the copy ${our_hash:-none}"
    exit 1
fi

words="$(printf '%s\n' "$ours" | wc -l | tr -d ' ')"

if [ -z "$our_hash" ]; then
    echo "check-core-rule: FAIL: core_withheld.txt ($commit) has no '# sha256 <64 hex digits>' line"
    echo "  the $words words are equal; fn credential_like in $1 has sha256 $their_hash"
    exit 1
fi

if [ "$their_hash" != "$our_hash" ]; then
    echo "check-core-rule: FAIL: the function changed, not only its words: fn credential_like in $1"
    echo "  the $words words are equal, the text of the function is not the one the copy ($commit) was taken from"
    echo "  sha256 of the function in core: $their_hash"
    echo "  sha256 in core_withheld.txt:    $our_hash"
    echo "  read core's function, change screen::core_withholds to match it, then update the copy"
    exit 1
fi

echo "check-core-rule: ok: $words words and the function's sha256 equal ($commit) in $1"
exit 0
