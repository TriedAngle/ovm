# Conformance tests

Small ECMAScript-semantics regression tests. Each script asserts the
behavior required by the spec and throws on mismatch; an uncaught
exception makes the process exit non-zero.

Run:

    cargo build --release -p vm
    for f in tests/conformance/*.js; do
        target/release/ovm "$f" || echo "FAIL $f"
    done
