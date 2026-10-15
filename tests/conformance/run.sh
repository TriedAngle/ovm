#!/usr/bin/env bash
# Differential conformance runner: every test runs in ovm and in a
# reference engine (node by default); exit codes and stdout must match.
#
# Known divergences (one open issue each) are listed in
# known-divergences.txt; they are reported without failing the run.
# A test listed as divergent that matches the reference engine is
# reported as XPASS -- remove it from the manifest once its issue is
# fixed.  Any unlisted divergence fails the run.
#
# Env:
#   OVM               path to the ovm binary (default target/release/ovm)
#   REFERENCE_ENGINE  reference jsvm (default node)

set -u
cd "$(dirname "$0")/../.."

OVM="${OVM:-target/release/ovm}"
REF="${REFERENCE_ENGINE:-node}"

if [ ! -x "$OVM" ]; then
    echo "ovm binary not found at $OVM (build with: cargo build --release -p vm)" >&2
    exit 2
fi

MANIFEST=tests/conformance/known-divergences.txt

known() {  # known <filename>
    [ -f "$MANIFEST" ] && grep -q "^$1 " "$MANIFEST"
}

fail=0
for t in tests/conformance/*.js; do
    name=$(basename "$t")

    ovm_out=$("$OVM" "$t" 2>/dev/null)
    ovm_code=$?
    ref_out=$("$REF" "$t" 2>/dev/null)
    ref_code=$?

    if [ "$ovm_code" = "$ref_code" ] && [ "$ovm_out" = "$ref_out" ]; then
        if known "$name"; then
            echo "XPASS  $name (matches $REF now -- remove from known-divergences.txt)"
        else
            echo "PASS   $name"
        fi
    elif known "$name"; then
        echo "KNOWN  $name (see $MANIFEST)"
    else
        echo "FAIL   $name"
        echo "  ovm: exit $ovm_code, stdout: ${ovm_out:-<empty>}"
        echo "  ref: exit $ref_code, stdout: ${ref_out:-<empty>}"
        fail=1
    fi
done

exit $fail
