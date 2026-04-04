from yahpo_gym import local_config, benchmark_set

BENCH_NAME = "lcbench"
TARGET_KEY = "loss"  

def make_benchmark(instance: str, data_path: str = "yahpo_data"):
    local_config.init_config()
    local_config.set_data_path(data_path)
    bench = benchmark_set.BenchmarkSet(BENCH_NAME)
    bench.set_instance(instance)
    return bench

def run_one(bench, cfg: dict, seed: int = 0):
    out = bench.objective_function(cfg)
    out = out[0]

    if isinstance(out, dict) and TARGET_KEY in out:
        score = float(out[TARGET_KEY][0])
    else:
        k = list(out.keys())[0] if isinstance(out, dict) else None
        score = float(out[k]) if k else float(out)

    return score, out