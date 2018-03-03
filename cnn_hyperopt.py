
import numpy as np
import datetime
import sys

import os
import errno
import yaml
import tensorflow as tf
import argparse
import subprocess
import glob

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

import sys

hyperopt_eval_num = 1


def hyperopt_search_space_from_gcloud_yaml(yaml_file_name):
    yaml_dict = yaml.load(open(yaml_file_name))
    yaml_hparams = yaml_dict['trainingInput']['hyperparameters']['params']
    search_space = {}
    is_int = {}
    if len(yaml_hparams) < 1:
        print("'params' is empty.")
        exit(1)
    for param in yaml_hparams:
        param_name = param['parameterName']
        param_type = param['type']
        if param_type == 'INTEGER':
            search_space[param_name] = hp.quniform(param_name, param['minValue'], param['maxValue'], 1)
            is_int[param_name] = True
        elif param_type == 'DOUBLE':
            search_space[param_name] = hp.uniform(param_name, param['minValue'], param['maxValue'])
            is_int[param_name] = False
        else:
            print("Unimplemented hyper param type '{}' used for parameter {}".format(param_type, param_name))
            exit(1)

    objective_tensor = yaml_dict['trainingInput']['hyperparameters']['hyperparameterMetricTag']
    return search_space, is_int, objective_tensor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num-trials',
        help='number of trials in run in hyperparameter optimization',
        type=int,
        default=1
    )

    parser.add_argument(
        '--config',
        help='path to hptuning yaml file',
        required=True
    )

    parser.add_argument(
        '--jobs-dir',
        help='directory to store all trial logs and exported information',
        required=True
    )

    parser.add_argument(
        'task_args',
        help='additional args to pass to task.py',
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        '--',
        dest='task_args',
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    task_args = args.task_args[1:]

    # def evaluate_classifier(n_estimators, learning_rate, subsample, colsample_bytree, max_depth, min_child_weight,
    #                         num_splits=2):
    #     global eval_num, max_evals
    #
    #     print("\n------------------------------------------------------")
    #     print("Trial {} of {}".format(eval_num, max_evals))
    #     print(
    #         "Evaluating - n_estimators: {} learning_rate: {}, subsample: {}, colsample_bytree: {}, max_depth: {}, min_child_weight: {}".format(
    #             n_estimators, learning_rate, subsample, colsample_bytree, max_depth, min_child_weight))
    #
    #     print("Working on Fold (of {}): ".format(num_splits), end='')
    #     sys.stdout.flush()
    #
    #     skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=7)
    #     k_aucs = np.full(num_splits, -1.0)
    #     i = 0
    #     for train_index, test_index in skf.split(X, Y):
    #         print(i + 1, end='')
    #         sys.stdout.flush()
    #
    #         # print("TRAIN:", train_index, "TEST:", test_index)
    #         X_train, X_test = X[train_index], X[test_index]
    #         Y_train, Y_test = Y[train_index], Y[test_index]
    #
    #         const_param = {
    #             'gpu_id': 0,
    #
    #             'tree_method': 'gpu_exact',  # <-- either use THIS
    #
    #             # 'max_bin': 16,            # <-- or THIS
    #             # 'tree_method': 'gpu_hist',  # <--
    #
    #             'n_estimators': n_estimators,
    #             'learning_rate': learning_rate,
    #             'subsample': subsample,
    #             'colsample_bytree': colsample_bytree,
    #             'max_depth': max_depth,
    #             'min_child_weight': min_child_weight
    #         }
    #
    #         model = XGBClassifier(**const_param)
    #
    #         model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], eval_metric='auc', verbose=False)
    #
    #         evals_result = model.evals_result()
    #         # print("\nEVALS RESULT: {}".format(evals_result))
    #         k_aucs[i] = evals_result['validation_0']['auc'][-1]
    #         print("(auc was {}) ".format(k_aucs[i]), end='')
    #         sys.stdout.flush()
    #
    #         i += 1
    #
    #     mean_auc = np.mean(k_aucs)
    #     print("\n\n{} Mean Auc".format(mean_auc))
    #     print("\n------------------------------------------------------\n")
    #     eval_num += 1
    #     return mean_auc
    #
    #
    # print('Running Parameter Search...')
    search_space, is_int, objective_tensor_name = hyperopt_search_space_from_gcloud_yaml(args.config)

    def objective(params):
        global hyperopt_eval_num

        params = {key: int(value) if is_int[key] else value for key, value in params.items()}

        job_dir = "{}/{}".format(args.jobs_dir, hyperopt_eval_num)

        command = "python -m trainer.task" \
                  + ''.join(" \\\n --{} {}".format(key, value) for key, value in params.items()) \
                  + ' \\\n '\
                  + ' '.join(element for element in task_args) \
                  + ' \\\n --job-dir ' + job_dir

        print('\n\n\n\n\n')
        print("Trial {}".format(hyperopt_eval_num))
        print(command)
        print()

        result = subprocess.run(command, shell=True)

        hyperopt_eval_num += 1

        if result.returncode == 0:

            eval_summary_files = glob.glob('{}/eval_intermediate_export/events.out.tfevents*'.format(job_dir))
            if len(eval_summary_files) == 0:
                print('Couldn\'t find any eval summary files')
                exit(1)

            objective_summaries = []
            for e in tf.train.summary_iterator(eval_summary_files[-1]):
                for v in e.summary.value:
                    if v.tag == objective_tensor_name:
                        # print(v.tag, v.simple_value)
                        objective_summaries.append(v.simple_value)
            print("\nFINAL RMSE: {}\n".format(objective_summaries[-1]))

            return {
                'loss': objective_summaries[-1],  # what we're trying to minimize
                'status': STATUS_OK,
                # -- store other results like this
                'params': params,
                'fdsfsd': objective_summaries[-1]
            }
        else:

            print("Removing Checkpoints")
            if job_dir is None or len(job_dir) == 0:
                print("\n\n\nERROR, JOB_DIR is empty or None")
                exit(1)

            for filename in glob.glob("{}/model*".format(job_dir)):
                os.remove(filename)
            for filename in glob.glob("{}/checkpoint".format(job_dir)):
                os.remove(filename)

            print("TRIAL FAILED\n")
            return {
                'status': STATUS_FAIL,
                # -- store other results like this
                'params': params
            }

    print("\n\n\n\nMonitor With: tensorboard --logdir {}".format(args.jobs_dir))

    trials = Trials()
    best_params = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=args.num_trials,  # how many parameter combinations we want to try
                trials=trials)

    print('\n\n\n\n\nDone, Evaluating Final Results')

    print("\n\nBEST RESULT:")
    print(best_params)

    final_results = objective(best_params)
    print(final_results)
    print("Best Loss: {}".format(final_results['loss']))
    print("\nExiting.")



