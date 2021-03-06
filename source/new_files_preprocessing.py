import os
import sys
import shutil

import pandas as pd
import numpy as np

from settings import local_dir, joint_labels, sensor_labels, segment_labels, matfiles_descriptions_path



source_dir = local_dir + "NMB\\"

measurements = ["jointAngle", "sensorMagneticField", "velocity", "angularVelocity"]

# Note: not including 'D10v1', 'D4v3', as ambulatory subjects

subjects = ["D2v1", "D3v1", "D4v1", "D4v2", "D5v1", "D5v2", "D5v3", "D6v1", "D6v2", "D6v3", "D7v1", "D7v2", "D7v3",
            "D9v1", "D9v2", "D9v3", "D11v1", "D11v2", "D11v3", "D12v1", "D12v2", "D12v3", "D14v1", "D14v2",
            "D14v3", "D15v1", "D15v2", "D15v3", "D16v1", "D17v1", "D17v2", "D17v3", "D18v1", "D18v2", "D18v3", "D19v1",
            "D20v1", "D20v2", "D21v1", "HC3v1", "HC4v1", "HC5v1"]
subjects_6mw = ["D2v1", "D3v1", "D4v1", "D4v2", "D5v1", "D5v2",  "D6v1", "D7v1", "D9v1", "D11v1", "D12v1",
                "D14v1", "D15v1","D17v1","D18v1", "D19v1", "D20v1", "HC1v1", "HC2v1", "HC3v1", "HC4v1", "HC5v1",
                "HC6v1", "HC7v1", "HC8v1", "HC9v1", "HC10v1", "HC11v1", "HC12v1", "HC13v1"]
output_types = ["overall", "acts"]

hc_subjects_leave_out = ["HC6v1", "HC7v1", "HC8v1", "HC9v1", "HC10v1", "HC11v1", "HC12v1", "HC13v1"]
rnn_fixed_args = f"--seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --no_testset --batch --combined"





#   #   #   #   #   New MPS to Run  #   #   #   #   #


def add_predictions_metadata(mps, aggregated=False, measures=None):

    if measures is None:
        measures = measurements

    model_pred_path = f"..\\documentation\\model_predictions\\model_predictions_mps{mps}.csv"
    df = pd.read_csv(model_pred_path)

    if not aggregated:
        overall_avg_abs_diffs = round(np.mean(df.iloc[:, 8].values), 2)
        for measure in measures:
            # Selects the rows where the measurement in question is tested
            measure_df = df.loc[df["Measurements tested"] == f"['{measure}']"]
            # Gets the average absolute difference between true and pred for measurement in question
            avg_abs_diffs = round(np.mean(measure_df.iloc[:, 8].values), 2)
            df = df.append({"Short file name": f"Average absolute difference between true and predicted ({measure})",
                            "Source dir": avg_abs_diffs}, ignore_index=True)

        df = df.append({"Short file name": f"Average absolute difference between true and predicted (overall)",
                        "Source dir": overall_avg_abs_diffs}, ignore_index=True)

    else:
        try:
            overall_avg_abs_diffs = round(np.mean(df.iloc[:, 11].values), 2)
        except TypeError:
            overall_avg_abs_diffs = round(np.mean(df.iloc[:, 8].values), 2)
        overall_aggregate_avg_abs_diffs = round(np.mean(df.iloc[:, 13].values), 2)
        for measure in measures:
            # Selects the rows where the measurement in question is tested
            measure_df = df.loc[df["Measurements tested"] == f"['{measure}']"]
            # Gets the average absolute difference between true and pred for measurement in question
            try:
                avg_abs_diffs = round(np.mean(measure_df.iloc[:, 11].values), 2)
            except TypeError:
                avg_abs_diffs = round(np.mean(measure_df.iloc[:, 8].values), 2)
            df = df.append({"Short file name": f"Average absolute difference between true and predicted ({measure})",
                            "Source dir": avg_abs_diffs}, ignore_index=True)

        df = df.append({"Short file name": f"Average absolute difference between true and predicted (overall)",
                        "Source dir": overall_avg_abs_diffs}, ignore_index=True)
        for measure in measures:
            # Selects the rows where the measurement in question is tested
            measure_df = df.loc[df["Measurements tested"] == f"['{measure}']"]
            # Gets the average absolute difference between true and pred for measurement in question
            avg_abs_diffs = round(np.mean(measure_df.iloc[:, 13].values), 2)
            df = df.append({"Short file name": f"Average aggregate absolute difference between true and predicted ({measure})",
                            "Source dir": avg_abs_diffs}, ignore_index=True)

        df = df.append({"Short file name": f"Average aggregate absolute difference between true and predicted (overall)",
                        "Source dir": overall_aggregate_avg_abs_diffs}, ignore_index=True)

    print("Writing predictions metadata...")
    df.to_csv(model_pred_path, index=False)


def mps_zero(measures, leave_out_subjects):
    """
        Same as MPS 1 but with only using 6MW matfiles so as to ensure that the models are valid to be used (i.e. they
        are as good as claimed by previous summer's work)
    """

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python rnn.py 6MW {measure} all overall {rnn_fixed_args.split('--combined')[0]} "
                      f"--leave_out={leave_out_subject} "
                      f"--model_path=MPS0_{measure}_{leave_out_subject}_overall ")

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python model_predictor.py 6MW {measure} {leave_out_subject} "
                      f"--no_testset --batch --mps=0")

    add_predictions_metadata("0")


def mps_zerofive(measures, leave_out_subjects):
    """
        Same as MPS 0 but using the single-acts scores rather than the overall score
    """

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python rnn.py 6MW {measure} all acts {rnn_fixed_args.split('--combined')[0]} "
                      f"--leave_out={leave_out_subject} "
                      f"--model_path=MPS0_{measure}_{leave_out_subject}_acts")

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python model_predictor.py 6MW {measure} {leave_out_subject} "
                      f"--no_testset --batch --mps=0 --combine_preds --extra_str=-5")

    add_predictions_metadata("0-5", aggregated=True)


def mps_one(measures, leave_out_subjects):
    """
        Comparing the use of various raw measurements (jointAngle, sensorMagneticField, velocity, jointVelocity) to
        predict on all left-out subjects when trained / evaluated on only 'v1' data from NMB
        and max 200k lines per file, using only the 'overall' output type
    """

    for measure in measures:
        for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
            os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS1_{measure}_{leave_out_subject}_overall "
                      f"--leave_out_version=v2,v3 "
                      f"--max_lines_per_file=200000")
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--leave_out_version=v2,v3 --no_testset --batch --mps=1")


def mps_onefive(measures, leave_out_subjects):
    """
        Same as MPS one except using the non-combined files rather than the combined files (so x20 the files
        but x20 fewer max lines per file, i.e. 10k rather than 200k). Note: still assessing on combined files
    """

    leave_out_subjects = [los for los in leave_out_subjects if "v1" in los]

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args.split(' --combined')[0]} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS1-5_{measure}_{leave_out_subject}_overall "
                      f"--leave_out_version=v2,v3 "
                      f"--max_lines_per_file=10000")

    for measure in measures:
        for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--leave_out_version=v2,v3 --no_testset --batch --mps=1-5 --use_combined")

    add_predictions_metadata("1-5")




def mps_onesevenfive(leave_out_subjects):
    """
        Same as MPS onefive except with using double max lines per file but balancing down; also only
        using the joinAngle measurement for the time being (to compare with the jointAngle from previous MPS which
        was the best measurement)
    """

    leave_out_subjects = [los for los in leave_out_subjects if "v1" in los]

    for leave_out_subject in leave_out_subjects:
        os.system(f"python rnn.py NMB jointAngle all overall {rnn_fixed_args.split(' --combined')[0]} "
                  f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                  f"--model_path=MPS1-75_jointAngle_{leave_out_subject}_overall "
                  f"--leave_out_version=v2,v3 --balance=down "
                  f"--max_lines_per_file=20000")

    for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
        os.system(f"python model_predictor.py NMB jointAngle {leave_out_subject} "
                  f"--leave_out_version=v2,v3 --no_testset --batch --mps=1-75 --use_combined")

    add_predictions_metadata("1-75")



def mps_two(leave_out_subjects):
    """
        Builds the same models as MPS1.5 except for acts output type and then assessing jointAngle specific
        measurement on combined predictions
    """

    leave_out_subjects = [los for los in leave_out_subjects if "v1" in los]

    for leave_out_subject in leave_out_subjects:
        os.system(f"python rnn.py NMB jointAngle all acts {rnn_fixed_args.split(' --combined')[0]} "
                  f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                  f"--model_path=MPS2_jointAngle_{leave_out_subject}_acts "
                  f"--leave_out_version=v2,v3 "
                  f"--max_lines_per_file=10000")

    for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
        os.system(f"python model_predictor.py NMB jointAngle {leave_out_subject} "
                  f"--leave_out_version=v2,v3 --no_testset --batch --mps=1-5,2 --use_combined --combine_preds")

    add_predictions_metadata("1-5,2", measures=["jointAngle"], aggregated=True)



def mps_three(measures, leave_out_subjects):
    """
        Builds a single models for each measurement with non of the 'v1' files left out and also using all the HC
        subjects now (but still not using any of the v2 or v3 files and not using some of the HC files) and assess these
        models on v2 and v3 files (i.e. so will be familiar with the subjects themselves but not their new version files)
    """

    for measure in measures:
        os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args.split(' --combined')[0]} "
                  f"--model_path=MPS3_{measure}_all_overall "
                  f"--leave_out_version=v2,v3 "
                  f"--max_lines_per_file=10000")

    for measure in measures:
        for leave_out_subject in [los for los in leave_out_subjects if "v1" not in los]:
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--no_testset --batch --mps=3 --use_combined --use_all")

    add_predictions_metadata("3", measures=measures)



def mps_four(leave_out_subjects):
    """
        Same as MPS three except only using jointAngle data and leaving out each of the subjects, one by one in turn
        (i.e. doing over ALL subjects in 'subjects', including the HC left out subjects and also 'v2' and 'v3'), using
        half the number of max lines per file (5k down from 10k), and assessing on combined files for each
        of the above subjects
    """


    for leave_out_subject in leave_out_subjects:
            os.system(f"python rnn.py NMB jointAngle all overall {rnn_fixed_args.split(' --combined')[0]} "
                      f"--leave_out={leave_out_subject} "
                      f"--model_path=MPS4_jointAngle_{leave_out_subject}_overall "
                      f"--max_lines_per_file=5000")


    for leave_out_subject in leave_out_subjects:
        os.system(f"python model_predictor.py NMB jointAngle {leave_out_subject} "
                  f"--no_testset --batch --mps=4 --use_combined")

    add_predictions_metadata("4", measures=["jointAngle"])



def mps_five(measures, leave_out_subjects):
    """
        Same as MPS three except only using NSAA files (but using all the lines per file since we don't use as many now)
        and over multiple measures
    """


    for measure in measures:
        for leave_out_subject in leave_out_subjects:
                os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args.split} "
                          f"--leave_out={leave_out_subject} "
                          f"--model_path=MPS5_{measure}_{leave_out_subject}_overall "
                          f"--inner_nsaa")

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--no_testset --batch --mps=5 --inner_nsaa")

    add_predictions_metadata("5", measures=measures)






def mps_five_old(measures, leave_out_subjects, out_types):
    """
        Similar MPS4 but with the addition of NSAA data along with NMB data (reducing max 10k to 5k lines) and testing
        over each measurement and output type (as the addition of a different input data type necessitates this), but
        otherwise keeping the same as MPS4 (with testing over every subject)
    """

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            for output_type in out_types:
                os.system(f"python rnn.py NSAA,NMB {measure} all {output_type} {rnn_fixed_args.split(' --combined')[0]} "
                          f"--leave_out={leave_out_subject} "
                          f"--model_path=MPS5_{measure}_{leave_out_subject}_{output_type} "
                          f"--max_lines_per_file=5000")

    for measure in measures:
        for leave_out_subject in leave_out_subjects:
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--no_testset --batch --mps=5 --use_combined --add_dir=NSAA --combine_preds")











def mps_oneeightfive(measures, leave_out_subjects):
    """
        Same as MPS one except with 1/4 lines per file as MPS 1 (might be able to avoid overtraining the RNNs this way)
    """

    for measure in measures:
        for leave_out_subject in [los for los in leave_out_subjects if "v1" in los]:
            os.system(f"python rnn.py NMB {measure} all overall {rnn_fixed_args} "
                      f"--leave_out={leave_out_subject},{','.join(hc_subjects_leave_out)} "
                      f"--model_path=MPS1_{measure}_{leave_out_subject}_overall "
                      f"--leave_out_version=v2,v3 "
                      f"--max_lines_per_file=50000")
            os.system(f"python model_predictor.py NMB {measure} {leave_out_subject} "
                      f"--leave_out_version=v2,v3 --no_testset --batch --mps=1")


def mps_onenine(leave_out_subjects):
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
                  f"--leave_out_version=v2,v3 --combine_preds --no_testset --batch  --mps=1,2")

def mps_four_old(leave_out_subjects, out_types):
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
                  f"--combine_preds --no_testset --batch --mps=4")


def mps_five_old(leave_out_subjects, out_types):
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
                  f"--add_dir=NSAA --combine_preds --no_testset --batch --mps=5")






#mps_zero(measurements, subjects_6mw)
#mps_zerofive(measurements, subjects_6mw)
#mps_one(measurements, subjects)
#mps_onefive(measurements, subjects)
#mps_onesevenfive(subjects)
#mps_two(subjects)
#mps_three(measurements, subjects)
#mps_four(subjects)

#add_predictions_metadata("1")




def file_renamer():
    os.system("python file_renamer.py NMB")


def ext_raw_measures(dir, measures):
    measure_str = ",".join(measures)
    os.system(f"python ext_raw_measures.py {dir} all {measure_str}")


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



def create_nmb_nsaa_dir():
    # Gets the list of the file names in the NMB directory that correspond to NSAA activity files
    df = pd.read_csv(f"..{matfiles_descriptions_path}", delimiter=";")
    df['Description'] = df['Description'].str.lower()
    filtered_df = df[df["Description"].str.contains("nsaa|north", na=False)]
    nsaa_file_names = filtered_df["File Name"].tolist()
    nsaa_file_names_partial = ["-".join(fn.split(".")[0].split("-")[:2]) for fn in nsaa_file_names]

    # Creates the NSAA subdirectory if it doesn't exist already
    nsaa_dir = f"{source_dir}NSAA\\"
    if not os.path.exists(nsaa_dir):
        os.mkdir(nsaa_dir)

    # Finds any matching files in the 'NMB' directory and copies them to an inner 'NSAA' directory
    for fn in os.listdir(source_dir):
        if any(nsaa_fn in fn for nsaa_fn in nsaa_file_names_partial):
            shutil.copy(src=f"{source_dir}{fn}", dst=f"{nsaa_dir}{fn}")



create_nmb_nsaa_dir()