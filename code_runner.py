import subprocess
import sys
import os
import argparse

# The instances identified for the benchmark
INSTANCES = ["3945", "7593", "34539", "146212", "168329"]
SEEDS = [0, 1, 2]

# Map of optimizer names to their script paths
OPTIMIZERS = {
    "asha": "optimizers/asha_optimizer.py",
    "cats": "optimizers/cats.py",
    "catsplus": "optimizers/catsplus.py",
    "catsplus_v2": "optimizers/catsplus_v2.py",
    "tpe": "optimizers/optuna_tpe.py",
    "random": "optimizers/random_search.py"
}

def run_experiment(opt_name, n_evals, instances, seeds):
    """
    Runs a specific optimizer across specified instances and seeds.
    """
    script_path = OPTIMIZERS.get(opt_name)
    if not script_path:
        print(f"Error: Optimizer '{opt_name}' not found.")
        return

    for inst in instances:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f">>> STARTING: {opt_name.upper()} | Instance: {inst} | Seed: {seed}")
            print(f"{'='*60}")
            
            cmd = [
                sys.executable,
                script_path,
                "--instance", str(inst),
                "--seed", str(seed),
                "--n_evals", str(n_evals)
            ]
            
            # Ensure 'runner' module is found by setting PYTHONPATH
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
            
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"ERROR running {opt_name} on instance {inst}, seed {seed}: {e}")
            except KeyboardInterrupt:
                print("\nExecution stopped by user.")
                sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Solution Runner for Budgeted HPO Benchmark")
    parser.add_argument("optimizer", choices=["all"] + list(OPTIMIZERS.keys()), 
                        help="Which optimizer to run ('all' runs everything)")
    parser.add_argument("--n_evals", type=int, default=20, 
                        help="Number of evaluations (default: 20)")
    parser.add_argument("--instances", nargs="+", default=INSTANCES, 
                        help="Instances to run on (default: all project instances)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, 
                        help="Seeds to run on (default: 0, 1, 2)")
    
    args = parser.parse_args()
    
    opts_to_run = list(OPTIMIZERS.keys()) if args.optimizer == "all" else [args.optimizer]
    
    for opt in opts_to_run:
        run_experiment(opt, args.n_evals, args.instances, args.seeds)

    print("\n" + "#"*60)
    print("ALL EXPERIMENTS COMPLETED.")
    print("#"*60)
