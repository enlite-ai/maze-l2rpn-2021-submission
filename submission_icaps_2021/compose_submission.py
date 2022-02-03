"""Script for composing a maze submission handling import and file transfer"""

import argparse
import os
import shutil
from typing import Optional

from maze.utils.bcolors import BColors

from submission_icaps_2021.check_your_submission import evaluate_submission as check_your_submission
from submission_icaps_2021.get_diff_import import get_missing_packages
from submission_icaps_2021.maze_agent.submission import EXPERIMENT_PATH

MAZE_SUBMISSION_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), 'maze_agent')
EXPERIMENT_DIR_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), './../experiment_data')
PACKAGES_FOLDER = 'packages/'

EXP_FILES_TO_COPY = ['.pkl', '.pt', '.yaml']
EXCLUDE_PACKAGES = ['pytest', 'pygraphviz']


def parse_args():
    """Argument parser for composing submission"""
    parser = argparse.ArgumentParser(description="Zip and check codalab.")
    parser.add_argument("--target_dir", default='/tmp/maze_l2rpn_submission',
                        help="Path where the submission folder should be build", type=str)
    parser.add_argument("--target_agent_file", required=False,
                        help="The Agent file you want to submit. If experiment path is given, the default maze "
                             "structured torch agent will be used", default=MAZE_SUBMISSION_PATH_DEFAULT,
                        type=str)
    parser.add_argument("--copy_maze", default=True, help='Specify if you would like to copy maze or download it from'
                                                          ' git', type=bool)
    parser.add_argument("--experiment_dir", help='Specify the experiment directory you would like to use (path to '
                                                 'training artifacts)', type=str, required=False,
                        default=EXPERIMENT_DIR_PATH_DEFAULT)
    parser.add_argument("--check_submission", default=True, help='Specify weather to check the submission after '
                                                                 'setting it up', type=bool)
    parser.add_argument('--agent_name', default=None, type=str)
    return parser.parse_args()


def compose_submission(target_dir_path: str, target_agent_path: Optional[str], experiment_path: Optional[str],
                       check_submission: bool):
    """Compose the submission and optionally check it on the local validation set.

    :param target_dir_path: Path to where the submission folder should be build.
    :param target_agent_path: Optional The Agent file you want to submit. If experiment path is given, the default maze
        structured torch agent will be used.
    :param experiment_path: Optional The abs path to the experiment output with the hydra config and state_spaces.
    :param check_submission: Specify weather to check the submission after setting it up.
    """

    # Remove directory and copy template directory
    print("[0]: setting up target dir")
    if os.path.exists(target_dir_path):
        shutil.rmtree(target_dir_path)

    # Copy the maze agent
    assert not os.path.exists(target_dir_path)
    shutil.copytree(target_agent_path, target_dir_path)

    submission_packages_path = os.path.join(target_dir_path, PACKAGES_FOLDER)

    # get all necessary python packages
    print("[1]: installing packages")
    num_package_installed = 0
    while True:
        missing_modules = get_missing_packages(target_dir_path)
        missing_modules = dict(filter(lambda item: item[0] not in EXCLUDE_PACKAGES, missing_modules.items()))
        missing_modules = dict(filter(lambda item: not os.path.exists(os.path.join(submission_packages_path, item[0])),
                                      missing_modules.items()))
        missing_modules = dict(filter(lambda item: not os.path.exists(
            os.path.join(submission_packages_path, item[0] + '.py')), missing_modules.items()))
        if len(missing_modules) == 0:
            break

        for package_name, package_path in missing_modules.items():
            if package_name in EXCLUDE_PACKAGES:
                continue

            if os.path.isdir(package_path):
                model_source_path = os.path.join(package_path, package_name)
            else:
                model_source_path = package_path
            module_target_path = os.path.join(submission_packages_path, package_name)
            if os.path.exists(module_target_path) or os.path.exists(module_target_path + '.py'):
                continue

            print(f'\t[1.{num_package_installed}]: installing {package_name}')
            if os.path.exists(model_source_path) and os.path.isdir(model_source_path):
                shutil.copytree(model_source_path, module_target_path)
            elif os.path.exists(model_source_path) and os.path.isfile(model_source_path):
                shutil.copyfile(model_source_path, module_target_path)
            elif os.path.exists(model_source_path + '.py'):
                shutil.copyfile(model_source_path + '.py', module_target_path + '.py')
            else:
                BColors.print_colored(f'Package {package_name} could not be found at {model_source_path}',
                                      color=BColors.WARNING)

            num_package_installed += 1

    if experiment_path is not None:
        print("[2]: copy experiment files")
        target_exp_path = os.path.join(target_dir_path, EXPERIMENT_PATH)
        if os.path.exists(target_exp_path):
            shutil.rmtree(target_exp_path)

        # Copy hydra configs
        shutil.copytree(experiment_path, target_exp_path)

    if check_submission:
        print("[3]: checking submission")
        check_your_submission(target_dir_path)

    print("[-]: done")


if __name__ == '__main__':
    args = parse_args()

    main_target_path = os.path.abspath(args.target_dir)
    main_agent_path = None if args.target_agent_file is None else os.path.abspath(args.target_agent_file)
    main_experiment_path = None if args.experiment_dir is None else os.path.abspath(args.experiment_dir)

    compose_submission(target_dir_path=main_target_path, target_agent_path=main_agent_path,
                       experiment_path=main_experiment_path,
                       check_submission=args.check_submission)
