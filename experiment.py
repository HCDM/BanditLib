import argparse
import os

def ingest_config(infilepath):
    pass

def run_simulation(config_filepath):
    os.system('python Simulation.py --config {}'.format(config_filepath))

def print_summary():
    pass

def run_experiment():
    # For each algorithm
        # Load info
        # For each perturbation setup
            # Run simulation
            # Check difference in number of arms
            # Append to file
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--config', dest='config', help='yaml config file')
    args = parser.parse_args()
    ingest_config(args.config)
    run_simulation(args.config)
    run_experiment()
    print_summary()
