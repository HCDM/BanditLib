import argparse
import json
import matplotlib.pyplot as plt
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

def rename_article_history_file(srcpath, dstpath):
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

def print_summary(arm_changes_filepath):
    print('Summary')
    with open(arm_changes_filepath, 'r') as infile:
        print(infile.read())


def compute_arm_diffs(orig_arm_file, new_arm_file):
    with open(orig_arm_file, 'r') as infile:
        orig_aids = json.load(infile)
    
    with open(new_arm_file, 'r') as infile:
        new_aids = json.load(infile)

    num_diffs = 0
    for time in range(len(orig_aids)):
        time_key = str(time)
        for user_id in orig_aids[time_key]:
            if orig_aids[time_key][user_id] != new_aids[time_key][user_id]:
                num_diffs += 1
    return num_diffs

def clear_changes(arm_changes_filepath):
    with open(arm_changes_filepath, 'w') as infile:
        pass

def save_changes(arm_changes_filepath, resample_round, resample_change, num_arm_changes):
    with open(arm_changes_filepath, 'a+') as outfile:
        outfile.write('{},{},{}\n'.format(resample_round, resample_change, num_arm_changes))

def display_arm_changes(orig_arm_file, new_arm_file):
    with open(orig_arm_file, 'r') as infile:
        orig_aids = json.load(infile)
    
    with open(new_arm_file, 'r') as infile:
        new_aids = json.load(infile)

    changes = []
    num_diffs = 0
    for time in range(len(orig_aids)):
        time_key = str(time)
        for user_id in orig_aids[time_key]:
            if orig_aids[time_key][user_id] != new_aids[time_key][user_id]:
                num_diffs += 1
        changes.append(num_diffs)

    plt.plot(range(len(orig_aids)), changes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('--config', dest='config', help='yaml config file')
    parser.add_argument('--clean', dest='clean', help='delete tmp folder before start', action='store_true')
    parser.add_argument('--load', dest='load', help='force everything to load in config', action='store_true')
    args = parser.parse_args()

    config_filepath = args.config
    config = read_config(config_filepath)
    experiment = config['experiment']
    tmp_dir = experiment['tmp_directory'] if experiment.has_key('tmp_directory') else 'tmp'


    # Delete tmp
    if args.clean:
        delete_tmp_directory(tmp_dir)

    # Create tmp
    create_tmp_directory(tmp_dir)
    tmp_exp_config_path = os.path.join(tmp_dir, 'tmp.exp.yaml')

    # Force load
    if args.load:
        config = set_config_only_loads(config)
        write_config(tmp_exp_config_path, config)

    # Run once to generate arms, pools, etc. as specified
    config = prepare_config(config)
    write_config(tmp_exp_config_path, config)
    if does_config_require_saving(config):
        # Save as necessary
        run_simulation(tmp_exp_config_path)

    config = set_config_only_loads(config)
    write_config(tmp_exp_config_path, config)
    art_sel_hist_file = experiment['article_selection_file']
    original_art_sel_hist_file = os.path.join(tmp_dir, 'article_selection_history_original.json')
    if not os.path.isfile(original_art_sel_hist_file):
        run_simulation(tmp_exp_config_path)
        rename_article_history_file(art_sel_hist_file, original_art_sel_hist_file)

    resample_round_interval = experiment['resample']['round_interval']
    resample_change = experiment['resample']['change']
    arm_changes_filepath = experiment['arm_changes_file']
    clear_changes(arm_changes_filepath)
    save_changes(arm_changes_filepath, 'Round', 'Change', 'DiffArms')

    for n in range(0, config['general']['testing_iterations'] + 1, resample_round_interval):
        config = set_resample(config, n, resample_change)
        write_config(tmp_exp_config_path, config)
        run_simulation(tmp_exp_config_path)

        arm_changes = compute_arm_diffs(original_art_sel_hist_file, art_sel_hist_file)
        save_changes(arm_changes_filepath, n, resample_change, arm_changes)
        display_arm_changes(original_art_sel_hist_file, art_sel_hist_file)

    plt.title('Cumulative arm changes with ({}) change'.format(resample_change))
    plt.show()
    print_summary(arm_changes_filepath)
