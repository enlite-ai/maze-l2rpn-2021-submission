import os

import pkg_resources

from submission_icaps_2021.get_info_import import process_file_import


def get_top_packages(package):
    try:
        top_levels = package.get_metadata("top_level.txt")
        top_levels = top_levels.split("\n")
        return top_levels[:-1]
    except KeyError:
        return None
    except FileNotFoundError:
        return None


def get_missing_packages(path_agent):
    folder = os.path.abspath(path_agent)
    all_modules = {}
    file_that_import = {}
    all_files = {}
    all_dirs = set()
    for root, dirs, files in os.walk(folder, topdown=True):
        dirs[:] = [d for d in dirs if d not in {"__pycache__"}]
        all_dirs.update(dirs)
        for file_ in files:
            if os.path.splitext(file_)[1] == ".pyc":
                continue
            mod_this_file = process_file_import(root, file_)
            all_files[file_] = mod_this_file
            for el in mod_this_file:
                if el not in all_modules:
                    all_modules[el] = []
                all_modules[el].append(mod_this_file[el])
                if el not in file_that_import:
                    file_that_import[el] = []
                file_that_import[el].append(os.path.join(root, file_))

    installed_packages = [(get_top_packages(package), package.module_path) for package in
                          pkg_resources.working_set]
    installed_packages = filter(lambda x: x[0] is not None, installed_packages)
    installed_packages = list(installed_packages)

    missing_packages = dict()
    # format all packages
    for packages, path in installed_packages:
        if path is None:
            continue

        for package in packages:
            if package.strip() and package in all_modules:
                missing_packages[package] = path
    return missing_packages
