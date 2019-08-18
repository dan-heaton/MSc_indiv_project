import argparse
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from settings import results_path, local_dir, model_pred_path
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument("choice",
                    help="Choice of graph creator mode. Must be one of: 'trues_preds' to call the 'plot_trues_preds()' "
                         "function, 'model_preds_altdirs' to call the 'plot_model_preds_altdirs()' function, "
                         "'model_preds_trues_preds' for 'plot_model_preds_trues_preds()', 'model_preds_single_acts' "
                         "for 'plot_model_preds_single_acts()', or 'rnn_results' to call the 'plot_rnn_results() function.")
parser.add_argument("arg_one", nargs="?", default=None,
                    help="Arg based on 'choice' that is specified. If choice='trues_preds', 'arg_one' is the name of "
                         "the file to load from 'RNN_outputs' (not inc. file extension) to plot the trues and preds "
                         "from. If choice='model_pred_altdirs', 'arg_one' is the name of the source directory to use "
                         "for the rows to be selected from 'model_predictions.csv'. If choice='model_preds_trues_preds', "
                         "'arg_one' is the row number within 'model_predictions.csv' to start from to draw the true and "
                         "predicted values from (inclusive). If choice='model_preds_single_acts', 'arg_one' is the short "
                         "file name of the file of which to load the 'single act' rows. If choice='rnn_results', "
                         "'arg_one' is the start experiment number to load from 'RNN Results.xlsx'.")
parser.add_argument("arg_two", nargs="?", default=None,
                    help="Arg based on 'choice' that is specified. If choice='model_preds_altdirs', 'arg_two' is the "
                         "name of the model trained dirs to use for the rows to be selected from 'model_predictions.csv' "
                         "(comma separated). If choice='model_preds_trues_preds', 'arg_two' is the row number within "
                         "'model_predictions.csv' to end from to draw the true and predicted values from (inclusive). "
                         "If choice='rnn_results', 'arg_two' is the end experiment number to load from 'RNN Results.xlsx.")
parser.add_argument("out_type", nargs="?", default=None,
                    help="Only used when choice='rnn_results'. Name of metric of which we wish to plot the results."
                         " Must be one of 'acc', 'mse', 'mae', 'rmse', 'r2', 'ind_act', or 'all_act'.")
parser.add_argument("xaxis", nargs="?", default=None,
                    help="Only used when choice='rnn_results'. Values to be plotting on the x-axis. Must be one of "
                         "'seq_len' (for plotting sequence length along the x-axis), 'ft' (for plotting the different "
                         "file types along x-axis), 'seq_over' (for plotting sequence overlaps along the x-axis), "
                         "or 'features'.")
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
parser.add_argument("--batch", type=bool, nargs="?", const=True, default=False,
                    help="Option that is only set if the script is run from a batch file to access the external files "
                         "in a correct way.")
args = parser.parse_args()


choices = ["trues_preds", "model_preds_altdirs", "model_preds_trues_preds", "model_preds_single_acts", "rnn_results"]
if args.choice not in choices:
    print("First arg ('arg_one') must be one of 'trues_preds', 'model_preds_altdirs', or 'rnn_results'...")
    sys.exit()



def plot_trues_preds():
    """
        Loads the file, reads in the predicted and true values, plots them, shows it to the user,
        and saves the image to the 'Graphs' directory
    """
    try:
        trues_preds = pd.read_csv(local_dir + "output_files\\RNN_outputs\\" + args.arg_one + ".csv")
        trues, preds = trues_preds["Trues"], trues_preds["Predictions"]
        fig, ax = plt.subplots()
        ax.scatter(trues, preds, alpha=0.03)
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x)
        plt.title("Plot of true overall NSAA scores against predicted overall NSAA scores")
        plt.xlabel("True overall NSAA scores")
        plt.ylabel("Predicted overall NSAA scores")
        if args.save_img:
            plt.savefig("..\\documentation\\Graphs\\" + args.arg_one)
        plt.gcf().set_size_inches(10, 10)
        if not args.no_display:
            plt.show()
    except FileNotFoundError:
        print("First arg ('arg_one') must be the name of a file within 'RNN_outputs' and cannot load '" +
              local_dir + "output_files\\RNN_outputs\\" + args.arg_one + "' ...")
        sys.exit()



def plot_model_preds_altdirs():
    """
        Handles the case where we are graphing with the 'model_predictions.csv' file, i.e. the 'test_altdirs.py' output
    """
    model_preds = pd.read_csv(model_pred_path)
    #Select the relevant rows from the file based on the args that we passed into 'graph_creator.py'
    model_preds = model_preds.loc[(model_preds["Source dir"] == args.arg_one) &
                                  (model_preds["Model trained dir(s)"] == str(args.arg_two.split(",")))]
    model_train_dirs = args.arg_two.split(",")


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
        plt.title("'model_predictions.csv', Source dir = " + args.arg_one + ", Model trained dir(s) = " +
                  args.arg_two.split(",")[i] + "\n" +
                  "Plot of true overall NSAA scores against predicted overall NSAA scores")
        plt.xlabel("True overall NSAA scores")
        plt.ylabel("Predicted overall NSAA scores")
        plt.gcf().set_size_inches(10, 10)
        if args.save_img:
            plt.savefig("..\\documentation\\Graphs\\Model_predictions_" + args.arg_one + "_" + args.arg_two + "_" +
                        args.arg_two.split(",")[i] + "_true_pred_overall_NSAA")
        if not args.no_display:
            plt.show()
        plt.close()

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
                  "w/ " + args.arg_two.split(",")[i] + " model pred dir")
        plt.xlabel("Percentage of correctly predicted sequence D/HC labels for given file")
        plt.ylabel("Number of files")
        plt.gcf().set_size_inches(10, 10)
        if args.save_img:
            plt.savefig("..\\documentation\\Graphs\\Model_predictions_" + args.arg_one + "_" + args.arg_two + "_" +
                        args.arg_two.split(",")[i] + "_distrib_seq_labels")
        if not args.no_display:
            plt.show()
        plt.close()

    #Plots distributions of the percentages of correctly predicted single acts for each file
    for i, md_off in enumerate(model_dir_offsets):
        percents = [int(round(float(row[md_off+3].split("%")[0]))) for j, row in model_preds.iterrows()]
        plt.hist(percents, 100)
        plt.hist(percents, 100, histtype='step', cumulative=True)
        plt.title("Distribution of percentage of individual acts correctly predicted over files "
                  "w/ " + args.arg_two.split(",")[i] + " model pred dir")
        plt.xlabel("Percentage of correctly predicted sequence individual acts labels for given file")
        plt.ylabel("Number of files")
        plt.gcf().set_size_inches(10, 10)
        if args.save_img:
            plt.savefig("..\\documentation\\Graphs\\Model_predictions_" + args.arg_one + "_" + args.arg_two + "_" +
                        args.arg_two.split(",")[i] + "_distrib_perc_indiv_acts")
        if not args.no_display:
            plt.show()
        plt.close()

    #Computes and prints MAE between true and predicted NSAA scores over files
    for i, md_off in enumerate(model_dir_offsets):
        mae = round(mean_absolute_error(model_preds.iloc[:, md_off+8].tolist(),
                                        model_preds.iloc[:, md_off+9].tolist()), 2)
        print("MAE between true and predicted NSAA scores over files (" +
              args.arg_two.split(",")[i] + ") = " + str(mae))
    print("\n")

    #Computes and prints percentage of correct predicted file D/HC label
    for i, md_off in enumerate(model_dir_offsets):
        correct_labels = 0
        for j, row in model_preds.iterrows():
            correct_labels += 1 if row[md_off+4] == row[md_off+5] else 0
        print("Percentage of correct predicted file D/HC label (" + args.arg_two.split(",")[i] + ") = " +
              str(round(((correct_labels/len(model_preds))*100), 2)) + "%")
    print("\n")

    #Computes and prints MAE of percentage predicted wrong sequence D/HC classification over files
    for i, md_off in enumerate(model_dir_offsets):
        pred_percents = []
        for j, row in model_preds.iterrows():
            if row[md_off+4] == "D":
                pred_percents.append(float(row[md_off+6][:-1]))
            else:
                pred_percents.append(float(row[md_off+7][:-1]))
        mae = round(mean_absolute_error(pred_percents, [100.0 for k in range(len(pred_percents))]), 2)
        print("MAE of percentage predicted wrong sequence D/HC classification over files (" +
              args.arg_two.split(",")[i] + ") = " + str(mae))
    print("\n")

    #Computes and prints average percentage of single acts correctly predicted over files
    for i, md_off in enumerate(model_dir_offsets):
        pred_percents = [float(row[md_off+3][:-1]) for j, row in model_preds.iterrows()]
        mean_perc = round(np.mean(pred_percents), 2)
        print("Average percentage of single acts correctly predicted over files (" +
              args.arg_two.split(",")[i] + ") = " + str(mean_perc) + "%")



def plot_model_preds_trues_preds():
    """
        Plots the true and predicted NSAA values for the selected rows (given by 'arg_one' and 'arg_two', inclusive)
        on a graph, with true values plotted on 'x-axis' and predicted values along the 'y-axis'
    """
    model_preds = pd.read_csv("..\\" + model_pred_path) if args.batch else pd.read_csv(model_pred_path)

    try:
        start_row, end_row = int(args.arg_one), int(args.arg_two)
    except ValueError:
        print("First two args ('arg_one' and 'arg_two') must be integers to select the row range "
              "of 'model_predictions.csv...")
        sys.exit()

    model_preds = model_preds.iloc[start_row-2:end_row-1, :]
    trues = model_preds.iloc[:, 6].astype(float).tolist()
    preds = model_preds.iloc[:, 7].astype(float).tolist()
    fig, ax = plt.subplots()
    ax.scatter(trues, preds, alpha=0.3)
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x)
    plt.title("Plot of true overall NSAA scores against predicted overall NSAA scores")
    plt.xlabel("True NSAA scores")
    plt.ylabel("Predicted NSAA scores")
    if args.save_img:
        if args.batch:
            plt.savefig("..\\..\\documentation\\Graphs\\" + "model_preds_trues_preds_" + args.arg_one + "_" + args.arg_two)
        else:
            plt.savefig("..\\documentation\\Graphs\\" + "model_preds_trues_preds_" + args.arg_one + "_" + args.arg_two)
    plt.gcf().set_size_inches(10, 10)
    if not args.no_display:
        plt.show()



def plot_model_preds_single_acts():
    """
        Plots 3 subplots where the act number (between 1 and 17) are along the x-axis and the results of the 3 output
        types for each activity is plotted along the y-axis.
    """
    model_preds = pd.read_csv(model_pred_path)
    #Select the relevant rows from the file based on the args that we passed into 'graph_creator.py'
    model_preds = model_preds.loc[(model_preds["Short file name"].str.contains(args.arg_one)) &
                                  (model_preds["Short file name"].str.contains("\(act"))]

    #Creates a dictionary with the act number as the keys and the value being a single list of 3 values, with values of
    #percent of acts correctly predicted, percent of predicted D/HC (i.e. correct) sequences, and predicted overall NSAA
    act_dict = {line[1].loc["Short file name"].split(" ")[-1][:-1] : [] for line in model_preds.iterrows()}
    for line in model_preds.iterrows():
        act_num = line[1].loc["Short file name"].split(" ")[-1][:-1]
        act_dict[act_num].append(float(line[1].loc["Percent of acts correctly predicted"][:-1]))
        if line[1].loc["True 'D/HC Label'"] == "D":
            act_dict[act_num].append(float(line[1].iloc[11][:-1]))
        else:
            act_dict[act_num].append(float(line[1].iloc[12][:-1]))
        act_dict[act_num].append(abs(line[1].loc["Predicted 'Overall NSAA Score'"] - line[1].loc["True 'Overall NSAA Score'"]))

    #Constructs the subplots for the 3 types of output type metrics for the single act numbers using the
    #entries in 'act_dict'
    fig, axes = plt.subplots(nrows=3)
    sc1 = axes[0].plot([act for act in act_dict], [act_dict[act][0] for act in act_dict])
    sc2 = axes[1].plot([act for act in act_dict], [act_dict[act][1] for act in act_dict])
    sc3 = axes[2].plot([act for act in act_dict], [act_dict[act][2] for act in act_dict])
    axes[0].set(xlabel="Act Number", ylabel="Percent of acts \ncorrectly predicted")
    axes[0].set_ylim(bottom=0, top=100)
    axes[1].set(xlabel="Act Number", ylabel="Percent of predicted \ncorrect D/HC sequences")
    axes[1].set_ylim(bottom=0, top=100)
    axes[2].set(xlabel="Act Number", ylabel="Absolute error between true\n and predicted overall NSAA score")
    axes[2].set_ylim(bottom=0, top=34)

    plt.suptitle("Plots of '" + args.arg_one + "' single activities performance for 'acts', 'dhc', and "
                                              "'overall' output types")
    plt.subplots_adjust(hspace=0.5, top=0.95, bottom=0.05)
    plt.gcf().set_size_inches(18.5, 10.5)
    if args.save_img:
        plt.savefig("..\\documentation\\Graphs\\Model_predictions_" + args.arg_one + "_single_acts")
    if not args.no_display:
        plt.show()
    plt.close()



def plot_rnn_results():
    out_type_map = {"acc": "Test Accuracy", "mse": "Mean Squared Error", "mae": "Mean Absolute Error",
                   "rmse": "Root Mean Squared Error", "r2": "R^2 Score",
                   "ind_act": "Individual Activity Accuracy", "all_act": "All Activities Accuracy"}
    xaxis_choices = ["seq_len", "ft", "seq_over", "features"]

    #Load the table of RNN results and find the minimum and maximum experiment numbers to use in argument validation
    results_table = pd.read_excel(results_path)
    results_table_min_max_vals = [int(min(results_table["Experiment Number"].values)),
                                  int(max(results_table["Experiment Number"].values))]

    #Checks there is a valid starting experiment number within the table
    if int(args.arg_one) >= results_table_min_max_vals[0]:
        start_pos = int(args.arg_one)
    else:
        print("First arg ('arg_one') must be >= the available experiment numbers...")
        sys.exit()

    #Checks there is a valid ending experiment number within the table
    if int(args.arg_two) <= results_table_min_max_vals[1] and int(args.arg_two) > start_pos:
        end_pos = int(args.arg_two)
    else:
        print("Second arg ('arg_two') must be <= the available experiment numbers and > the 'arg_one' arg...")
        sys.exit()

    #Ensures that the choice of metric to plot in the graph is a valid choice
    if args.out_type in out_type_map:
        choice = args.out_type
    else:
        print("Third arg ('out_type') must be one of the available choices for plotting metrics...")
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

    #Appends the measurement names, sequence lengths, and required results (based on the 'out_type' argument) from the
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
            if out_type_map[choice] == j.split(" = ")[0]:
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
        plt.title("Plot of " + str(out_type_map[choice]) + " and sequence length for experiments " +
                  str(start_pos) + " to " + str(end_pos))
        plt.xlabel("Sequence length")
        plt.ylabel(out_type_map[choice])
        plt.legend()

    elif xaxis_choice == "ft":
        plt_list = [[], []]
        for dir, measure, result in zip(dirs, measures, results):
            if result:
                dir_measure = measure if dirs.count(dirs[0]) == len(dirs) else dir + " : " + measure
                plt_list[0].append(dir_measure)
                plt_list[1].append(result)
        plt.plot(plt_list[0], plt_list[1], marker="o")
        plt.title("Plot of " + str(out_type_map[choice]) + " and dir/file_type for experiments " +
                  str(start_pos) + " to " + str(end_pos))
        plt.xlabel("Source directory and file type")
        plt.xticks(rotation=10, fontsize=5)
        plt.ylabel(out_type_map[choice])

    elif xaxis_choice == "seq_over":
        plt.plot(seq_overs, results, marker="o")
        plt.title("Plot of " + str(out_type_map[choice]) + " and sequence overlap for experiments " +
                  str(start_pos) + " to " + str(end_pos))
        plt.xlabel("Sequence overlap")
        plt.ylabel(out_type_map[choice])

    elif xaxis_choice == "features":
        plt.plot(features, results, marker="o")
        plt.title("Plot of " + str(out_type_map[choice]) + " and number of features for experiments " +
                  str(start_pos) + " to " + str(end_pos))
        plt.xlabel("Number of features")
        plt.ylabel(out_type_map[choice])

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



if args.choice == "trues_preds":
    plot_trues_preds()
elif args.choice == "model_preds_altdirs":
    plot_model_preds_altdirs()
elif args.choice == "model_preds_trues_preds":
    plot_model_preds_trues_preds()
elif args.choice == "model_preds_single_acts":
    plot_model_preds_single_acts()
elif args.choice == "rnn_results":
    plot_rnn_results()