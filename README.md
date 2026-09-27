# ovm
Default build
    cargo build --release -p vm

Fast build (nightly toolchain):

    cargo +nightly build --release -p vm --features fast

Run:
    target/release/ovm script.js

Minor-GC stress (collect on every allocation; much slower, catches GC bugs):
    cargo build --release -p vm --features stress-minor-gc

Combine with the fast interpreter:
    cargo +nightly build --release -p vm --features fast,stress-minor-gc
