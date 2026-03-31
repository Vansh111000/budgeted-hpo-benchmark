from yahpo_gym import local_config, benchmark_set 
local_config.init_config() 
local_config.set_data_path("yahpo_data") 
# points to the folder we cloned 

bench = benchmark_set.BenchmarkSet("lcbench")
bench.set_instance("3945")
cfg = bench.config_space.sample_configuration(1).get_dictionary() 
out = bench.objective_function(cfg) 
print("CFG:", cfg) 
print("OUT:", out)
print("DONE")