# -*- coding: utf-8 -*-
# pylint: disable=C0111,C0103,R0205
# !./synthesizer/bin/python

from d_evaluation_metrics.main import evaluate
from e_report_generation.main import generate_report

# file_path = "amatis/train/train.csv"
file_path = "amatis/33float/33float.csv"
# expenses.csv"

# result_status, result_message = overview_analysis(file_path)
# print("MODULE A -> ", result_message, result_status)
#
# result_status, result_message = preprocess(file_path)
# print("MODULE B -> ", result_message, result_status)
#
# result_status, result_message = generate_data(file_path)
# print("MODULE C -> ", result_message, result_status)
#
# result_status, result_message = evaluate(file_path)
# print("MODULE D -> ", result_message, result_status)

result_status, result_message = generate_report(file_path)
print("MODULE E -> ", result_message, result_status)
