#!/usr/bin/env bash
# check-core-rule.sh <core checkout>
#
# Axel's copy of core's withholding rule (crates/axel/src/core_withheld.txt) must hold the
# same words as the array in `fn credential_like` of the given Synaps core checkout.
# Takes the string literals of that array, sorts them, and compares them with the file's.
#   exit 0  equal
#   exit 1  different (the difference is printed), or core has no such array any more:
#           someone then writes the new mirror
#   exit 2  wrong usage, or a file is missing
# Only reads. Run by the foreman on a core change that touches the function, and by
# cutover.sh against the checkout the cutover binary was built from.

set -uo pipefail
export LC_ALL=C

[ $# -eq 1 ] || { echo "usage: check-core-rule.sh <core checkout>" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
copy="$here/crates/axel/src/core_withheld.txt"
core="$1/crates/agent-core/src/core/context_archive.rs"

[ -f "$copy" ] || { echo "check-core-rule: no $copy" >&2; exit 2; }
[ -f "$core" ] || { echo "check-core-rule: no $core" >&2; exit 2; }

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

ours="$(grep -v '^#' "$copy" | grep -v '^[[:space:]]*$' | sort)"
commit="$(head -n 1 "$copy")"

if [ -z "$theirs" ]; then
    echo "check-core-rule: FAIL: no array of string literals in fn credential_like of $core"
    echo "  core's rule has changed shape; the copy ($commit) needs a new mirror"
    exit 1
fi

if [ "$theirs" = "$ours" ]; then
    echo "check-core-rule: ok: $(printf '%s\n' "$ours" | wc -l | tr -d ' ') words equal ($commit) in $1"
    exit 0
fi

echo "check-core-rule: FAIL: core_withheld.txt ($commit) differs from fn credential_like in $1"
echo "  lines with < are only in core, lines with > are only in the copy:"
diff <(printf '%s\n' "$theirs") <(printf '%s\n' "$ours") | grep '^[<>]' | sed 's/^/  /'
exit 1
