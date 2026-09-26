// Minimal CLI runner for the vendored Octane suites: feed this file after
// base.js and the benchmark files, e.g.
//   ovm benchmarks/octane/base.js benchmarks/octane/deltablue/deltablue.js \
//      benchmarks/octane/run.js

var runner = {
  NotifyStart: function (name) {
    print("Running " + name + " ...");
  },
  NotifyStep: function (name) {
    // individual benchmark within a suite finished
  },
  NotifyError: function (name, error) {
    print("ERROR: " + name + ": " + error);
  },
  NotifyResult: function (name, score) {
    print(name + ": " + score);
  },
  NotifyScore: function (score) {
    print("----");
    print("Score: " + score);
  }
};

BenchmarkSuite.RunSuites(runner);
