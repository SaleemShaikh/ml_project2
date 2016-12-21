"""
run.py
Script to generate the best performance.

If you encounter any issue regarding running the file, please
go to project2/evaluation_fcn.py to set project path to the folder
containing 'project2' package.


"""
import os
import sys

import project2.evaluation_fcn as eval_fcn


def produce_best_result():
    eval_fcn.execute()

if __name__ == '__main__':
    produce_best_result()