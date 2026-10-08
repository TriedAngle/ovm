#!/usr/bin/env bash
# Wall-time benchmark harness for the OVM interpreters and QuickJS.
#
#   benchmarks/bench.sh <match-binary> <become-binary> [quickjs-binary] [node-binary]
#
# The node binary runs jitless (--jitless) with a print->console.log shim
# for the Octane runner's print().
#
# - fannkuch / nbody: the SunSpider files are top-level scripts far too
#   short to time directly, so the harness wraps the vendored source in a
#   function and calls it AMPLIFY times from one generated runner file.
#   The source is parsed exactly once, the body runs AMPLIFY times, and
#   VM startup is paid once per measurement (negligible against the
#   seconds of guest work). Every contender runs the identical file.
#   Reports the median wall time of RUNS measurements.
# - deltablue: Octane is time-boxed (~2s: 1s warmup + 1s measured) and
#   reports a normalized score (iterations/ms), so wall time is
#   meaningless and no amplification is needed: the score IS the metric.
#   Median score of RUNS measurements. The ovm binaries take the three
#   files as arguments (they join them into one script); QuickJS takes
#   them as `-I` includes plus the runner as the main file.
#
# Results are printed as a markdown table.

set -euo pipefail

MATCH="${1:?usage: bench.sh <match-binary> <become-binary> [quickjs-binary] [node-binary]}"
BECOME="${2:?usage: bench.sh <match-binary> <become-binary> [quickjs-binary] [node-binary]}"
QJS="${3:-}"
NODE="${4:-}"

RUNS=7
AMPLIFY=50
root="$(cd "$(dirname "$0")/.." && pwd)"
fan_js="$root/benchmarks/fannkuch/fannkuch.js"
nb_js="$root/benchmarks/nbody/nbody.js"
db_js=(
  "$root/benchmarks/octane/base.js"
  "$root/benchmarks/octane/deltablue/deltablue.js"
  "$root/benchmarks/octane/run.js"
)

# wrap `src` in a function called AMPLIFY times: one parse, n executions
make_runner() { # src outfile
  {
    echo "function __bench_body() {"
    cat "$1"
    echo "}"
    echo "for (var __i = 0; __i < $AMPLIFY; __i++) __bench_body();"
  } > "$2"
}

fan_file="$(mktemp -t fannkuch.XXXXXX).js"
nb_file="$(mktemp -t nbody.XXXXXX).js"
trap 'rm -f "$fan_file" "$nb_file"' EXIT
make_runner "$fan_js" "$fan_file"
make_runner "$nb_js" "$nb_file"

# Median wall time (microseconds) of RUNS executions, timed from a single
# Python process (one clock domain). Args: bin, then the script argv.
time_median_us() {
  RUNS=$RUNS python3 - "$@" <<'EOF'
import os, subprocess, sys, time
bin, args = sys.argv[1], sys.argv[2:]
runs = int(os.environ["RUNS"])
times = []
for _ in range(runs):
    t0 = time.perf_counter_ns()
    subprocess.run([bin, *args], stdout=subprocess.DEVNULL, check=True)
    times.append((time.perf_counter_ns() - t0) // 1000)
times.sort()
print(times[(runs - 1) // 2])
EOF
}

score_median() { # bin, then the script argv
  RUNS=$RUNS python3 - "$@" <<'EOF'
import os, re, subprocess, sys
bin, args = sys.argv[1], sys.argv[2:]
runs = int(os.environ["RUNS"])
scores = []
for _ in range(runs):
    out = subprocess.run([bin, *args], capture_output=True, text=True, check=True).stdout
    scores.append(float(re.search(r"^DeltaBlue: (\S+)", out, re.M).group(1)))
scores.sort()
print(scores[(runs - 1) // 2])
EOF
}

fmt_us() { awk -v us="$1" 'BEGIN { printf "%.2f", us / 1000000 }'; }
ratio()  { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a / b }'; }

declare -a BINS=("$MATCH" "$BECOME")
declare -a NAMES=("match_loop" "become")
if [[ -n "$QJS" ]]; then BINS+=("$QJS"); NAMES+=("quickjs"); fi
if [[ -n "$NODE" ]]; then
  # node runs exactly one script file per invocation: the shim plus the
  # three deltablue files are joined into one script
  node_all="$(mktemp -t node_all.XXXXXX).js"
  { echo "globalThis.print = console.log;"; cat "${db_js[@]}"; } > "$node_all"
  node_wrap="$(mktemp -t node_wrap.XXXXXX)"
  { echo "#!/bin/sh"; echo "exec '$NODE' --jitless '$node_all'"; } > "$node_wrap"
  chmod +x "$node_wrap"
  node_run="$(mktemp -t node_run.XXXXXX)"
  { echo "#!/bin/sh"; echo "exec '$NODE' --jitless \"\$1\""; } > "$node_run"
  chmod +x "$node_run"
  trap 'rm -f "$fan_file" "$nb_file" "$node_all" "$node_wrap" "$node_run"' EXIT
  BINS+=("$node_wrap"); NAMES+=("node_jitless")
fi

fan=(); nb=(); db=()
for i in "${!BINS[@]}"; do
  b="${BINS[$i]}"
  runner="$b"
  if [[ -n "$NODE" && "$b" == "$node_wrap" ]]; then
    runner="$node_run"
  fi
  fan+=("$(time_median_us "$runner" "$fan_file")")
  nb+=("$(time_median_us "$runner" "$nb_file")")
  if [[ "$b" == *qjs* ]]; then
    db+=("$(score_median "$b" -I "${db_js[0]}" -I "${db_js[1]}" "${db_js[2]}")")
  else
    db+=("$(score_median "$b" "${db_js[@]}")")
  fi
done

header="| benchmark | metric |"
sep="|---|---|"
for n in "${NAMES[@]}"; do header+=" $n |"; sep+="---|"; done
echo "$header"; echo "$sep"

fan_cells=""; nb_cells=""; db_cells=""
for t in "${fan[@]}"; do fan_cells+=" $(fmt_us "$t") |"; done
for t in "${nb[@]}"; do nb_cells+=" $(fmt_us "$t") |"; done
for t in "${db[@]}"; do db_cells+=" $t |"; done
echo "| fannkuch (x$AMPLIFY) | wall s (median $RUNS) |$fan_cells"
echo "| nbody (x$AMPLIFY) | wall s (median $RUNS) |$nb_cells"
echo "| deltablue | score (median $RUNS) |$db_cells"

ratios=""; base="${fan[1]}" # become is the reference column
for t in "${fan[@]}"; do ratios+=" $(ratio "$t" "$base")x |"; done
echo "| fannkuch ratio vs become | |$ratios"
ratios=""; base="${nb[1]}"
for t in "${nb[@]}"; do ratios+=" $(ratio "$t" "$base")x |"; done
echo "| nbody ratio vs become | |$ratios"
ratios=""; base="${db[1]}"
for t in "${db[@]}"; do ratios+=" $(ratio "$t" "$base")x |"; done
echo "| deltablue ratio vs become (higher = faster) | |$ratios"
