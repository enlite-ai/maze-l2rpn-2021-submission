import re
import os
import argparse

from submission_icaps_2021.utils import dict_installed_packages

DISCLAIMER_HEAD = """
DISCLAIMER: This is a tool to help you debug your imports faster. This might not give 100% accurate results. 
Keep in mind this is a facility to help you debug and NOT a file that always say only right things.
"""

DISCLAIMER_FOOT = """
DISCLAIMER: It is not because something is written in this log that this something is true.
Use with care, and keep in mind it is an help to assist you in debugging.
If automatic debugging was possible 100% of the time we would have simplified the submission process.

Don't forget to read the detailed error log for more informations

Only full python log can help you debug with 100% accuracy: 
Make sure to check the errors printed locally if running with "check_your_submission.py" or checking the "View ingestion output log" on codalab.
"""


DEFAULT_MODEL_DIR = 'example_submission/submission'


dict_import = {}  # key (path, file), value: list of all imports


def get_local_modules_from_li(li_imports):
    res = {}
    for el in li_imports:
        # first remove the preceeding "from" and "import" and the "as XXX" at the end
        cleaned = re.sub("(^\s*((from)|(import))\s*)|(as\s+[a-zA-Z0-9]+\s*$)", "", el)
        cleaned = cleaned.rstrip().lstrip()
        # grep the parent package
        parent, *childs = re.split("[\s\.]", cleaned)
        # child = re.sub("^{}".format(parent), "", cleaned)
        if not parent in dict_installed_packages:
            if not parent in res:
                res[parent] = []
            res[parent].append(el)
    return res


def process_file_import(root, file):
    this_f = os.path.join(root, file)
    name_, ext_ = os.path.splitext(this_f)
    res = {}
    if ext_ == ".py":
        with open(this_f, "r") as f:
            li = f.readlines()
        li = [el.rstrip().lstrip() for el in li]
        li = [el for el in li if re.match("^((from [a-zA-Z0-9])|(import [a-zA-Z0-9]))", el) is not None]
        modules = get_local_modules_from_li(li)
        for el in modules:
            if el not in res:
                res[el] = []
            res[el] = (modules[el], this_f)
    return res


def debug_imports(path_agent):
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

    for imported_module_name in all_modules:
        print_head = False
        HEARDER = "There might be an import error due to import of \"{}\"".format(imported_module_name)
        if re.match("(submission|\.)", imported_module_name) is None:
            print_head = True
            print(HEARDER)
            # error: import missing a "."
            for (wrong_imports, file) in all_modules[imported_module_name]:
                for wrong_import in wrong_imports:
                    good_import = re.sub("{}[^$]".format(imported_module_name),
                                         ".{} ".format(imported_module_name),
                                         wrong_import)

                    print("\t * Possible ERROR detected in the imports. Try replacing:"
                          "\n\t   \t\t{}\n\t   by\n\t   \t\t{}\n\t   in the file \"{}\"\n\t"
                          "might help you solve this error, but it might be a terrible advice. "
                          "Have a look there anyway :-)"
                          "".format(wrong_import, good_import, file))

        module_name = re.sub("^(\.|submission)", "", imported_module_name)
        if imported_module_name not in all_files and imported_module_name not in all_dirs:
            # this is most likely an error
            if not print_head:
                print_head = True
                print(HEARDER)
            print("\t * Possible ERROR could not locate the file \"{0}.py \" (this could also be a folder \"{0}/\")"
                  "".format(module_name))
        elif imported_module_name not in all_files and imported_module_name in all_files:
            # this is most likely due to this script not recursing on sub directories
            # it's a tool to help debug, not a python import module.
            if not print_head:
                print_head = True
                print(HEARDER)
            print("\t * WARNING could not locate the file \"{0}.py \" (this could also be a folder \"{0}/\")"
                  "".format(module_name))

        if print_head:
            print()


def cli():
    parser = argparse.ArgumentParser(description="Zip and check codalab.")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="Path of the model you want to submit.")
    return parser.parse_args()


def main(input_dir):
    debug_imports(input_dir)


if __name__ == "__main__":
    args = cli()

    print(DISCLAIMER_HEAD)
    main(args.model_dir)
    print(DISCLAIMER_FOOT)
