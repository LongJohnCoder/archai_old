import subprocess
import argparse
import os
from pathlib import Path
import shutil

from FastAutoAugment.common.common import common_init, get_config_common

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--scripts', type=str,
                        default='scripts/random/cifar_search.py,scripts/random/cifar_eval.py',
                        help='scripts to run in order, comma separated')
    parser.add_argument('--exp_prefix', type=str, default='random',
                        help='Experiment prefix to use')
    args, extra_args = parser.parse_known_args()

    conf = common_init(use_args=True)

    last_output_file = None
    for script in args.scripts.split(','):
        script = os.path.expandvars((os.path.expanduser(script.strip())))
        experiment_name = args.exp_prefix + '_' + Path(script).stem

        # copy output of last experiment to this experiment's folder
        logdir = get_config_common()['logdir']
        assert logdir
        experiment_dir = os.path.join(logdir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        if last_output_file:
            shutil.copy2(last_output_file, experiment_dir)

        script =  os.path.abspath(script)
        print(f'Starting {script}...')
        result = subprocess.run(
            ['python', script,
             '--config', conf.config_filepath,
             '--config-defaults', conf.config_defaults_filepath,
             '--common.experiment_name', experiment_name
            ])
        print(f'Script {script} returned {result.returncode}')

        if result.returncode != 0:
            exit(1)

        # TODO: need to remove file name hard coding
        last_output_file = os.path.join(experiment_dir, 'final_model_desc.yaml')
