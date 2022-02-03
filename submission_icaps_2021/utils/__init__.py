import os
import re
import sys
from .installed_packages import dict_installed_packages

UTILS_PATH = os.path.split(__file__)[0]

# reference path valid for the data
problem_dir = os.path.join(UTILS_PATH, 'ingestion_program_local/')
score_dir = os.path.join(UTILS_PATH, 'scoring_program_local/')
ref_data = os.path.join(UTILS_PATH, 'public_ref/')
ingestion_output = os.path.join(UTILS_PATH, 'logs/')
submission_program_default = os.path.join(os.path.split(UTILS_PATH)[0], "example_submissions")

input_data_check_dir = os.path.join(os.path.split(UTILS_PATH)[0], 'input_data_local/')
output_dir = os.path.join(UTILS_PATH, 'output/')


def ingestion_program_cmd(ingestion_program=problem_dir,
                          input_=input_data_check_dir,
                          output_submission=output_dir,
                          submission_program=submission_program_default,
                          python_exec=sys.executable):
    """
    Get the command to launch the submission program.

    This is one of the command that will be executed on the cloud to rank your submission. Of course with the
    data in your submission programm.

    /!\ DO NOT MODIFY THIS /!\

    Parameters
    ----------
    ingestion_program:
        Path of the ingestion program

    input_: ``str``
        Path of the input data

    output_submission:
        Path where the results will be stored

    submission_program
        path of the submissino program

    python_exec: ``str``
        path to the python executable you want to use.

    Returns
    -------
    cmd: ``str``

    """
    with open(os.path.join(ingestion_program, "metadata"), "r") as f:
        cmd_ = f.readlines()[0]
    if sys.platform.startswith('win'):
        # i am on windows, i need to fix all the "\\" in the paths...
        python_exec = re.sub("\\\\", "\\\\\\\\", python_exec)
        ingestion_program = re.sub("\\\\", "\\\\\\\\", ingestion_program)
        input_ = re.sub("\\\\", "\\\\\\\\", input_)
        output_submission = re.sub("\\\\", "\\\\\\\\", output_submission)
        submission_program = re.sub("\\\\", "\\\\\\\\", submission_program)
    cmd_ = re.sub("command: python ", "{} ".format(python_exec), cmd_)
    cmd_ = re.sub("\$ingestion_program", "{}".format(ingestion_program), cmd_)
    cmd_ = re.sub("\$input", "{}".format(input_), cmd_)
    cmd_ = re.sub("\$output", "{}".format(output_submission), cmd_)
    cmd_ = re.sub("\$submission_program", "{}".format(submission_program), cmd_)
    return cmd_


def scoring_program_cmd(scoring_program=score_dir,
                        input_scoring=output_dir,
                        output_scoring=output_dir,
                        python_exec=sys.executable):
    """
    Get the command to launch the scoring program.

    This is one of the command that will be executed on the cloud to rank your submission. Of course with the
    data in your submission programm.

    /!\ DO NOT MODIFY THIS /!\

    Parameters
    ----------
    scoring_program:
        Path of the srocing program

    input_scoring: ``str``
        Path of the input data (for the scoring program)

    output_scoring: ``str``
        Path of the ouput data (of the scoring program)

    python_exec: ``str``
        path to the python executable you want to use.

    Returns
    -------
    cmd: ``str``

    """
    with open(os.path.join(scoring_program, "metadata"), "r") as f:
        cmd_score = f.readlines()[0]
    if sys.platform.startswith('win'):
        # i am on windows, i need to fix all the "\\" in the paths...
        python_exec = re.sub("\\\\", "\\\\\\\\", python_exec)
        scoring_program = re.sub("\\\\", "\\\\\\\\", scoring_program)
        input_scoring = re.sub("\\\\", "\\\\\\\\", input_scoring)
        output_scoring = re.sub("\\\\", "\\\\\\\\", output_scoring)
    cmd_score = re.sub("command: python ", "{} ".format(python_exec), cmd_score)
    cmd_score = re.sub("\$program", "{}".format(scoring_program), cmd_score)
    cmd_score = re.sub("\$input", "{}".format(input_scoring), cmd_score)
    cmd_score = re.sub("\$output", "{}".format(output_scoring), cmd_score)
    return cmd_score
