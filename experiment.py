import argparse
import os
import shutil
import time
import yaml

def create_tmp_directory(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

def delete_tmp_directory(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

def read_config(infilepath):
    with open(infilepath, 'r') as infile:
        config = yaml.load(infile)
    return config

def write_config(outfilepath, config):
    with open(outfilepath, 'w') as outfile:
        outfile.write(yaml.dump(config))

def rename_article_history_csv(srcpath, dstpath):
    os.rename(srcpath, dstpath)

def prepare_config(config):
    config['general']['plot'] = False
    return config

def does_config_require_saving(config):
    if any([config['user']['save'],
            config['article']['save'],
            config['pool']['save'],
            config['reward']['noise']['save']]):
        return True
    for alg in config['alg']['specific']:
        if alg.has_key('noise_save') and alg['noise_save']:
            return True
    return False


def set_config_only_loads(config):
    config['user']['save'] = False
    config['user']['load'] = True
    
    config['article']['save'] = False
    config['article']['load'] = True
    
    config['pool']['save'] = False
    config['pool']['load'] = True
    
    config['reward']['noise']['save'] = False
    config['reward']['noise']['load'] = True
    
    for n in range(len(config['alg']['specific'])):
        config['alg']['specific'][n]['noise_save'] = False
        config['alg']['specific'][n]['noise_load'] = True

    return config

def set_resample(config, round, change):
    config['reward']['noise']['resample'] = {
        'round': round,
        'change': change
    }
    return config

def run_simulation(config_filepath):
    os.system('python Simulation.py --config {}'.format(config_filepath))

def print_summary():
    print('Summary')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--config', dest='config', help='yaml config file')
    parser.add_argument('--clean', dest='clean', help='delete tmp folder before start', action='store_true')
    args = parser.parse_args()

    config_filepath = args.config
    config = read_config(config_filepath)

    # Delete tmp
    if args.clean:
        delete_tmp_directory('tmp')

    # Create tmp
    create_tmp_directory('tmp')
    tmp_exp_config_path = os.path.join('tmp', 'tmp.exp.yaml')

    # Run once to generate arms, pools, etc. as specified
    config = prepare_config(config)
    write_config(tmp_exp_config_path, config)
    if does_config_require_saving(config):
        # Save as necessary
        run_simulation(tmp_exp_config_path)

    config = set_config_only_loads(config)
    if not os.path.isfile('tmp/article_selection_history_original.csv'):
        run_simulation(tmp_exp_config_path)
        rename_article_history_csv('tmp/article_selection_history.csv', 'tmp/article_selection_history_original.csv')

    for n in range(0, 100, 100):
        config = set_resample(config, n, 5)
        write_config(tmp_exp_config_path, config)
        run_simulation(tmp_exp_config_path)

    print_summary()
