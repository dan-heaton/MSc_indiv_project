import argparse
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from settings import results_path, local_dir

parser = argparse.ArgumentParser()
parser.add_argument("start_exp", nargs="?", default=None,
                    help="Experiment number in 'RNN Results.xlsx' of the first experiment (inclusive) to have "
                         "the graph consist.")
parser.add_argument("end_exp", nargs="?", default=None,
                    help="Experiment number in 'RNN Results.xlsx' of the last experiment (inclusive) to have "
                         "the graph consist.")
parser.add_argument("choice", nargs="?", default=None,
                    help="Name of metric of which we wish to plot the results. Must be one of 'acc', 'mse', "
                         "'mae', 'rmse', 'r2', 'ind_act', or 'all_act'.")
parser.add_argument("xaxis", nargs="?", default=None,
                    help="Values to be plotting on the x-axis. Must be one of 'seq_len' (for plotting sequence "
                         "length along the x-axis), 'ft' (for plotting the different file types along x-axis), "
                         "'seq_over' (for plotting sequence overlaps along the x-axis), or 'features'.")
parser.add_argument("--save_img", type=bool, nargs="?", const=True,
                    help="Save the image to the 'documentation\\Graphs' directory.")
parser.add_argument("--no_display", type=bool, nargs="?", const=True,
                    help="Specify if specifically don't want to open the plot before writing.")
parser.add_argument("--split_rows_overlap", type=bool, nargs="?", const=True,
                    help="Set if wish to divide up the rows to be plotted as 2 separate lines on the plot.")
parser.add_argument("--split_rows_disc", type=bool, nargs="?", const=True,
                    help="Set if wish to divide up the rows to be plotted as 2 separate lines on the plot.")
parser.add_argument("--x_log", type=bool, nargs="?", const=True,
                    help="Specify if wish to plot x-axis values in log scale (e.g. when plotting very large "
                         "sequence lengths.")
args = parser.parse_args()



#Only executed if the first argument is provided as a file name within 'RNN_outputs', whereupon it loads the file,
#reads in the predicted and true values, plots them, shows it to the user, and saves the image to the 'Graphs' directory
if "_" in args.start_exp and args.choice != "model_preds":
    try:
        trues_preds = pd.read_csv(local_dir + "output_files\\RNN_outputs\\" + args.start_exp + ".csv")
        trues, preds = trues_preds["Trues"], trues_preds["Predictions"]
        fig, ax = plt.subplots()
        ax.scatter(trues, preds, alpha=0.03)
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x)
        plt.title("Plot of true overall NSAA scores against predicted overall NSAA scores")
        plt.xlabel("True overall NSAA scores")
        plt.ylabel("Predicted overall NSAA scores")
        plt.savefig("..\\documentation\\Graphs\\" + args.start_exp)
        plt.gcf().set_size_inches(10, 10)
        plt.show()
    except FileNotFoundError:
        print("First arg ('start_exp') must be the name of a file within 'RNN_outputs' and cannot load '" +
              local_dir + "output_files\\RNN_outputs\\" + args.start_exp + "' ...")
        sys.exit()
    #Exit the program after plotting the graph, as remainder of the program expects to see other arguments present
    sys.exit()


#Handles the case where we are graphing with the 'model_predictions.csv' file, i.e. the 'test_altdirs.py' output
if args.choice == "model_preds":
    model_preds = pd.read_csv("..\\documentation\\model_predictions.csv")
    #Select the relevant rows from the file based on the args that we passed into 'graph_creator.py'
    model_preds = model_preds.loc[(model_preds["Source dir"] == args.start_exp) &
                                  (model_preds["Model trained dir(s)"] == ", ".join(args.end_exp.split(",")))]
    model_train_dirs = args.end_exp.split(",")


    cols = model_preds.columns.values.tolist()
    #Gets the index positions of the separator columns (i.e. the cols like '------ NSAA Predictions ------')
    model_dir_offsets = [cols.index(c) for c in cols if "------" in c]
    #Gets the index positions of all the 'true overall NSAA' and 'predict overall NSAA' columns
    tp_col_indices = np.ravel([[off + 8, off + 9] for off in model_dir_offsets]).tolist()
    #Select only the true/pred columns and reassemble them into a dictionary with keys corresponding to model dir names
    overall_tps = model_preds.iloc[:, tp_col_indices].values.T
    overall_tps = [[overall_tps[i], overall_tps[i+1]] for i in range(0, len(overall_tps), 2)]
    overall_tps_dict = {mtd: otp for mtd, otp in zip(model_train_dirs, overall_tps)}

    #Plots true overall NSAA scores against predicted overall NSAA scores
    for i, mtd in enumerate(overall_tps_dict):
        fig, ax = plt.subplots()
        ax.scatter(overall_tps_dict[mtd][0], overall_tps_dict[mtd][1], alpha=0.03)
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x)
        plt.title("'model_predictions.csv', Source dir = " + args.start_exp + ", Model trained dir(s) = " +
                  args.end_exp.split(",")[i] + "\n" +
                  "Plot of true overall NSAA scores against predicted overall NSAA scores")
        plt.xlabel("True overall NSAA scores")
        plt.ylabel("Predicted overall NSAA scores")
        plt.gcf().set_size_inches(10, 10)
        #plt.savefig("..\\documentation\\Graphs\\" + args.start_exp)
        plt.show()

    #Plots distributions of the percentages of correctly predicted sequence D/HC labels
    for i, md_off in enumerate(model_dir_offsets):
        percents = []
        for j, row in model_preds.iterrows():
            #Based on the value within the 'True D/HC label' column, pick either the percentage of predicted 'D'
            #sequences or predicted 'HC' sequences, remove the '%', round it to 0 decimal places, and append to
            #percents as an int
            if row[md_off+4] == "D":
                percents.append(int(round(float(row[md_off+6].split("%")[0]))))
            else:
                percents.append(int(round(float(row[md_off+7].split("%")[0]))))
        plt.hist(percents, 100)
        plt.hist(percents, 100, histtype='step', cumulative=True)
        plt.title("Distribution of percentage of correctly predicted sequence D/HC labels over files "
                  "w/ " + args.end_exp.split(",")[i] + " model pred dir")
        plt.xlabel("Percentage of correctly predicted sequence D/HC labels for given file")
        plt.ylabel("Number of files")
        plt.show()

    #Plots distributions of the percentages of correctly predicted single acts for each file
    for i, md_off in enumerate(model_dir_offsets):
        percents = [int(round(float(row[md_off+3].split("%")[0]))) for j, row in model_preds.iterrows()]
        plt.hist(percents, 100)
        plt.hist(percents, 100, histtype='step', cumulative=True)
        plt.title("Distribution of percentage of individual acts correctly predicted over files "
                  "w/ " + args.end_exp.split(",")[i] + " model pred dir")
        plt.xlabel("Percentage of correctly predicted sequence individual acts labels for given file")
        plt.ylabel("Number of files")
        plt.show()

    #Exit the program after plotting the graph, as remainder of the program expects to see other arguments present
    sys.exit()



choices_map = {"acc": "Test Accuracy", "mse": "Mean Squared Error", "mae": "Mean Absolute Error",
               "rmse": "Root Mean Squared Error", "r2": "R^2 Score",
               "ind_act": "Individual Activity Accuracy", "all_act": "All Activities Accuracy"}
xaxis_choices = ["seq_len", "ft", "seq_over", "features"]

#Load the table of RNN results and find the minimum and maximum experiment numbers to use in argument validation
results_table = pd.read_excel(results_path)
results_table_min_max_vals = [int(min(results_table["Experiment Number"].values)),
                              int(max(results_table["Experiment Number"].values))]

#Checks there is a valid starting experiment number within the table
if int(args.start_exp) >= results_table_min_max_vals[0]:
    start_pos = int(args.start_exp)
else:
    print("First arg ('start_exp') must be >= the available experiment numbers...")
    sys.exit()

#Checks there is a valid ending experiment number within the table
if int(args.end_exp) <= results_table_min_max_vals[1] and int(args.end_exp) > start_pos:
    end_pos = int(args.end_exp)
else:
    print("Second arg ('end_exp') must be <= the available experiment numbers and > the 'start_exp' arg...")
    sys.exit()

#Ensures that the choice of metric to plot in the graph is a valid choice
if args.choice in choices_map:
    choice = args.choice
else:
    print("Third arg ('choice') must be one of the available choices for plotting metrics...")
    sys.exit()

#Ensures that the choice of x-axis plotting element is a valid choice
if args.xaxis in xaxis_choices:
    xaxis_choice = args.xaxis
else:
    print("Fourth arg ('xaxis') must be one of the available choices for what is plotted along the x-axis...")
    sys.exit()

#Extracts the experiment rows that we are concerned with from the table of RNN results
start_index = results_table.index[results_table["Experiment Number"] == start_pos].tolist()[0]
end_index = results_table.index[results_table["Experiment Number"] == end_pos].tolist()[0]
exper_data = results_table.iloc[start_index:end_index+1, :]

#Appends the measurement names, sequence lengths, and required results (based on the 'choice' argument) from the
#relevant cells in each row to lists
dirs, measures, seq_lens, seq_overs, results, features = [], [], [], [], [], []
for i, row in exper_data.iterrows():
    dirs.append(row["'rnn' arguments"].split(" ")[2])
    measures.append(row["'rnn' arguments"].split(" ")[3])
    seq_lens.append(int(row["'rnn' arguments"].split("--seq_len=")[-1].split(" ")[0]))
    features.append(int(row[11].split(" ")[-1]))
    res_names = [j.split(" = ")[0] for j in row["Results"].split(", ")]
    res_res = [j.split(" = ")[1] for j in row["Results"].split(", ")]
    added_num = False
    for j in row["Results"].split(", "):
        if choices_map[choice] == j.split(" = ")[0]:
            num = j.split(" = ")[1]
            num = float(num[:-1]) if num.endswith("%") else float(num)
            results.append(num)
            added_num = True
    if not added_num:
        results.append(None)
    seq_overs.append(None if "seq_overlap" not in row["'rnn' arguments"]
                     else float(row["'rnn' arguments"].split(" ")[-1].split("=")[-1]))

if xaxis_choice == "seq_len":
    #Combine these measurements into a dictionary with each key being a measurement name and each value being a list
    #of two lists, with each of these sub-lists containing the sequence lengths and required results
    if args.split_rows_overlap:
        measures = np.ravel([["Same overlap" for i in range(len(measures)//2)],
                             ["Scaling overlap w/ seq_len" for i in range(len(measures)//2)]])
    elif args.split_rows_disc:
        measures = [[], []]
        for i, row in exper_data.iterrows():
            if "discard_prop" not in row["'rnn' arguments"]:
                measures[0].append("No discard_prop")
            else:
                measures[1].append("Discard prop")
        measures = [elem for inner in measures for elem in inner]
    df_dict = {measure: [[], []] for measure in measures}
    for i in range(len(measures)):
        df_dict[measures[i]][0].append(seq_lens[i])
        df_dict[measures[i]][1].append(results[i])
    #Creates a line from each entry in the table, with the label being the measurement name, the 'x' values being the
    #sequence lengths, and the 'y' values being the results values
    for measure in df_dict:
        plt.plot(df_dict[measure][0], df_dict[measure][1], label=measure, marker="o")
    plt.title("Plot of " + str(choices_map[choice]) + " and sequence length for experiments " +
              str(start_pos) + " to " + str(end_pos))
    plt.xlabel("Sequence length")
    plt.ylabel(choices_map[choice])
    plt.legend()

elif xaxis_choice == "ft":
    plt_list = [[], []]
    for dir, measure, result in zip(dirs, measures, results):
        if result:
            dir_measure = measure if dirs.count(dirs[0]) == len(dirs) else dir + " : " + measure
            plt_list[0].append(dir_measure)
            plt_list[1].append(result)
    plt.plot(plt_list[0], plt_list[1], marker="o")
    plt.title("Plot of " + str(choices_map[choice]) + " and dir/file_type for experiments " +
              str(start_pos) + " to " + str(end_pos))
    plt.xlabel("Source directory and file type")
    plt.xticks(rotation=10, fontsize=5)
    plt.ylabel(choices_map[choice])

elif xaxis_choice == "seq_over":
    plt.plot(seq_overs, results, marker="o")
    plt.title("Plot of " + str(choices_map[choice]) + " and sequence overlap for experiments " +
              str(start_pos) + " to " + str(end_pos))
    plt.xlabel("Sequence overlap")
    plt.ylabel(choices_map[choice])

elif xaxis_choice == "features":
    plt.plot(features, results, marker="o")
    plt.title("Plot of " + str(choices_map[choice]) + " and number of features for experiments " +
              str(start_pos) + " to " + str(end_pos))
    plt.xlabel("Number of features")
    plt.ylabel(choices_map[choice])

#Sets the scaling of the x-axis to logarithmic if the required argument is set
if args.x_log:
    plt.xscale("log")

#Only save the plot image to file (with a name based on the rows of data used and metric plotted) if argument is specified
if args.save_img:
    file_name = "Exp" + str(start_pos) + "-" + str(end_pos) + "_" + choice
    plt.savefig("..\\documentation\\Graphs\\" + file_name)

#Display the image to the user if the '--no_display' argument is not given
if not args.no_display:
    plt.gcf().set_size_inches(10, 10)
    plt.show()
