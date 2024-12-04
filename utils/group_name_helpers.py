from pathlib import Path
from pyterrier.io import read_results
import pyterrier
import pickle
from loguru import logger
import re
import settings as s
from slugify import slugify
import utils.repair_result_dataframe as rd


def get_runs_from_participants_touche21(desired_group_name):
    desired_group_name_slug = slugify(desired_group_name).lower()
    run_files = []

    # cycle through the incorporated files and perform the operations
    for group in Path(s.TOUCHE_DIR).iterdir():
        group_name = slugify(group.name).lower()

        if group_name == desired_group_name_slug:
            group_dir_with_runs = group.joinpath('output-deduplicated-with-copycat')
            if not group_dir_with_runs.exists():
                logger.info(f"Group {desired_group_name} has no deduplicated runs, trying to find runs in output directory")
                group_dir_with_runs = group.joinpath('output')

            for run_file in group_dir_with_runs.iterdir():
                run_file_name = slugify(run_file.name).lower()
                run_file_read = read_results(str(run_file)) # read run file with pyterrier
                repaired_run_file = rd.repair_touche_run(run_file_read)
                if repaired_run_file is not None:
                    run_files.append((run_file_name, repaired_run_file))
            return run_files
    return None


def get_runs_from_participants_touche20(desired_group_name):
    desired_group_name_slug = slugify(desired_group_name).lower()
    run_files = []

    # cycle through the incorporated files and perform the operations
    for group in Path(s.TOUCHE_DIR).iterdir():
        group_name = slugify(group.name).lower()

        if group_name == desired_group_name_slug:
            for run_dir in group.iterdir():
                run_file_name = slugify(run_dir.name).lower()
                for run_file in run_dir.iterdir():
                    run_file_read = read_results(str(run_file))
                    repaired_run_file = rd.repair_touche_run(run_file_read)  # read run file with pyterrier
                    if repaired_run_file is not None:
                        run_files.append((run_file_name, repaired_run_file))
            return run_files
    return None


def get_all_group_participants():
    directories = [slugify(d.name).lower() for d in Path(s.TOUCHE_DIR).iterdir() if d.is_dir()  and not d.name.startswith('.')]
    return directories

def get_best_run_for_participant(group_name,get_run_func=None,experiment_func=None):
    '''Experiment function determines which evaluation is used to determine the best run'''
    group_name_slug = slugify(group_name).lower()
    experiments_results = []
    runs = get_run_func(group_name)
    if len(runs) == 0:
        logger.error(f"No runs found for group {group_name}")
        return None,None,None,None
    best_run_nbr = None
    best_run = None
    best_run_result = - 1
    all_results = []
    for name,run in runs:
        result = experiment_func(group_name,run)
        all_results.append(result)
        experiments_results.append((name,result))
        if result > best_run_result:
            best_run = run
            best_run_result = result
            best_run_nbr = name
    avg = sum(all_results) / len(all_results)
    print(f"Average score for {group_name} is {avg}")
    print(f"Best score for {group_name} is {best_run_result}")
    #with open(results_path, 'wb') as f:
     #   pickle.dump((best_run_nbr,best_run,runs,experiments_results),f)
    return group_name_slug, best_run_nbr, best_run_result,best_run,experiments_results






