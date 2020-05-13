import os
import sys

import pandas as pd
import numpy as np

from settings import local_dir, joint_labels, sensor_labels, segment_labels



source_dir = local_dir + "NMB\\"

measurements = ["jointAngle", "sensorMagneticField", "velocity", "angularVelocity"]
subjects = ["D2v1", "D3v1", "D4v1", "D4v2", "D5v1", "D5v2", "D5v3", "D6v1", "D6v2", "D6v3", "D7v1", "D7v2", "D7v3",
            "D9v1", "D9v2", "D9v3", "D10v1", "D11v1", "D11v2", "D11v3", "D12v1", "D12v2", "D12v3", "D14v1", "D14v2",
            "D14v3", "D15v1", "D15v2", "D15v3", "D16v1", "D17v1", "D17v2", "D17v3", "D18v1", "D18v2", "D18v3", "D19v1",
            "D20v1", "D20v2", "D21v1", "HC3v1", "HC4v1", "HC5v1"]
output_types = ["overall", "acts"]

hc_subjects_leave_out = ["HC6v1", "HC7v1", "HC8v1", "HC9v1", "HC10v1", "HC11v1", "HC12v1", "HC13v1"]
rnn_fixed_args = f"--seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --no_testset --batch --combined"





#   #   #   #   #   New MPS to Run  #   #   #   #   #


def mps_one(measures, leave_out_subjects):
    """
        Comparing the use of various raw measurements (jointAngle, sensorMagneticField, velocity, jointVelocity) to
        predict on all left-out subjects when trained / evaluated on only 'v1' data from NMB
        and max 200k lines per file, using only the 'overall' output type
    """

    for measure in measures:
        for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
            """os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS1_{measure}_{leave_out_subject}_overall "
                      f"--leave_out_version=v2,v3 "
                      f"--max_lines_per_file=200000")"""
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--leave_out_version=v2,v3 --no_testset --batch")


def mps_two(leave_out_subjects):
    """
        Same as MPS 1, except builds the models only for the 'sensorMagneticField' meeasurement type and for the
        'acts' output type, and predicting subjects via models built in both MPS 1 and MPS 2 (i.e. wish to see how
        useful the '--combine_preds' optional argument is)
    """

    for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
        os.system(f"python rnn.py NMB sensorMagneticField all acts {rnn_fixed_args} "
                  f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                  f"--model_path=MPS2_sensorMagneticField_{leave_out_subject}_acts "
                  f"--leave_out_version=v2,v3 "
                  f"--max_lines_per_file=200000")
        os.system(f"python model_predictor.py NMB sensorMagneticField {leave_out_subject} "
                  f"--leave_out_version=v2,v3 --combine_preds --no_testset --batch")


def mps_three(leave_out_subjects):
    """
        Builds no new models, but instead evaluates all the non-version-1 files on models that had the subject
        left-out of the training set (i.e. same as 'model_predictor.py' in 'mps_one()' but for v2 and v3) and only
        for the sensorMagneticField, which will have been determined to be the best in 'mps_one()'
    """

    for leave_out_subject in [los for los in leave_out_subjects if "v1" not in los]:
        os.system(f"python model_predictor.py NMB sensorMagneticField {leave_out_subject} "
                  f"--combine_preds --no_testset --batch")


def mps_four(leave_out_subjects, out_types):
    """
        Compared with 'mps_one()', builds the same models as before but only used the sensorMagneticField measurement
        (which is determined to be the best measurement to use) and added in all version files (not just 'v1'),
        but reduced max lines per file from 200k to 100k
    """

    for leave_out_subject in leave_out_subjects:
        for output_type in out_types:
            os.system(f"python rnn.py NMB sensorMagneticField all {output_type} {rnn_fixed_args} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS4_{measure}_{leave_out_subject}_{output_type} "
                      f"--max_lines_per_file=100000")
        os.system(f"python model_predictor.py NMB sensorMagneticField {leave_out_subject} "
                  f"--combine_preds --no_testset --batch")


def mps_five(leave_out_subjects, out_types):
    """
        Addition of NSAA data along with NMB data (reducing max 100k to 50k lines), but otherwise keeping the same
        as the 'mps_three()'
    """

    for leave_out_subject in leave_out_subjects:
        for output_type in out_types:
            os.system(f"python rnn.py NSAA,NMB sensorMagneticField all {output_type} {rnn_fixed_args} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS5_{measure}_{leave_out_subject}_{output_type} "
                      f"--max_lines_per_file=50000")
        os.system(f"python model_predictor.py NMB sensorMagneticField {leave_out_subject} "
                  f"--add_dir=NSAA --combine_preds --no_testset --batch")




mps_one(["angularVelocity"], subjects, ["overall"])









def file_renamer():
    os.system("python file_renamer.py NMB")


def ext_raw_measures(measures):
    measure_str = ",".join(measures)
    os.system(f"python ext_raw_measures.py NMB all {measure_str}")


def combine_files(measures):
    headers = {"jointAngle": [f"({joint_label}) : ({axis}-axis)"
                              for joint_label in joint_labels for axis in ("X", "Y", "Z")],
               "sensorMagneticField": [f"({sensor_label}) : ({axis}-axis)"
                              for sensor_label in sensor_labels for axis in ("X", "Y", "Z")],
               "velocity": [f"({segment_label}) : ({axis}-axis)"
                              for segment_label in segment_labels for axis in ("X", "Y", "Z")],
               "angularVelocity": [f"({segment_label}) : ({axis}-axis)"
                            for segment_label in segment_labels for axis in ("X", "Y", "Z")]
               }

    for measure in measures:
        measurement_dir = source_dir + measure
        if not os.path.exists(measurement_dir + "\\combined"):
            os.mkdir(measurement_dir + "\\combined")
        file_names = [fn for fn in os.listdir(measurement_dir)]

        unique_subject_versions = sorted(list(set([fn.split("-")[0] for fn in file_names if fn != "combined"])))
        for usv in unique_subject_versions:
            total_lines = []
            for fn in [f for f in file_names if f != "combined"]:
                if usv in fn:
                    full_fn = measurement_dir + "\\" + fn
                    lines = pd.read_csv(full_fn, index_col=0, header=0).values.tolist()
                    total_lines += lines
            df = pd.DataFrame(total_lines, index=[usv for i in range(len(total_lines))], columns=headers[measure])
            new_file_name = f"{measurement_dir}\\combined\\{usv}-allfiles_{measure}.csv"
            print(f"Writing '{new_file_name}'...")
            df.to_csv(new_file_name)
