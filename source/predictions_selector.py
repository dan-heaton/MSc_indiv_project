import argparse
from settings import sub_dirs, model_pred_path
import pandas as pd
import sys


parser = argparse.ArgumentParser()
parser.add_argument("sfn", help="Specify the short file names of the subjects the user wishes to observe. Specify 'all' "
                                "if wish to select all possible rows given the other arguments.")
parser.add_argument("sd", help="Specify the source dir that the selected rows should be from in 'model_predictions'.")
parser.add_argument("--mtd", type=str, nargs="?", const=True, default=False,
                    help="Optional argument to filter rows based on the values within the 'Model trained dir(s). "
                         "Separate these 'altdirs' by commas.")
parser.add_argument("--best", type=str, nargs="?", const=True, default=False,
                    help="Optional argument to retrieve the best of the filtered rows. Provide two parts (separated by "
                         "a comma): first part to select the necessary output rows and being one of 'pacp' ('percent of "
                         "acts correctly predicted'), 'ppcs' ('percent of predicted correct sequences'), or 'overall "
                         "('rue..' and 'predicted overall NSAA score'). Second part being an integer of the number of "
                         "best performing rows according to this output metric.")
parser.add_argument("--worst", type=str, nargs="?", const=True, default=False,
                    help="Identical to '--best', except the second part of the arg is used to select the worst "
                         "performing rows according to the selected output metric.")
args = parser.parse_args()

#Loads in the 'model_predictions.csv' file
model_preds = pd.read_csv(model_pred_path)

metrics = ["pacp", "ppcs", "overall"]


#Filters the rows based on the 'sfn' arg if it isn't 'all'
if not args.sfn == "all":
    model_preds = model_preds.loc[model_preds["Short file name"].str.contains(args.sfn)]
    if len(model_preds.index) == 0:
        print("No row names matching the first arg ('" + args.sfn + "') for the 'Short file name' column....")
        sys.exit()

if args.sd + "\\" not in sub_dirs:
    print("Second arg ('" + args.sd + "') must be one of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'direct_csv', " +
          "'allmatfiles', or 'left-out'...")
    sys.exit()
else:
    model_preds = model_preds.loc[model_preds["Source dir"] == args.sd]

if args.mtd:
    altdirs = args.mtd.split(",")
    for altdir in altdirs:
        if altdir + "\\" not in sub_dirs:
            print("Optional arg '--mtd' must be one or more of '6minwalk-matfiles', '6MW-matFiles', 'NSAA', 'direct_csv', " +
                  "'allmatfiles', and/or 'left-out', with each separated by a comma...")
            sys.exit()
        else:
            model_preds = model_preds.loc[model_preds["Model trained dir(s)"].str.contains(altdir, na=False)]


selected_rows = [[], []]
best_worst_rows = [0, 0]
best_worst_metrics = [None, None]

#Given that the rows are now filtered correctly, display the remainder based on optionally-provided arguments
if args.best:
    metric, num_rows = args.best.split(",")
    if metric not in metrics:
        print("Optional arg '--best' must have first part (of two, split by comma) of one of 'pacp', 'pcs', "
              "or 'overall'...")
        sys.exit()
    else:
        best_worst_metrics[0] = metric
    try:
        best_worst_rows[0] = int(num_rows)
    except ValueError:
        print("Optional arg '--best' must have second part (of two, split by comma) to be an integer of the number of "
              "'n' best rows of filtered rows.")
        sys.exit()
if args.worst:
    metric, num_rows = args.worst.split(",")
    if metric not in metrics:
        print("Optional arg '--worst' must have first part (of two, split by comma) of one of 'pacp', 'pcs', "
              "or 'overall'...")
        sys.exit()
    else:
        best_worst_metrics[1] = metric
    try:
        best_worst_rows[1] = int(num_rows)
    except ValueError:
        print("Optional arg '--worst' must have second part (of two, split by comma) to be an integer of the number of "
              "'n' worst rows of filtered rows.")
        sys.exit()

#Enables us to select the correct column values for each of the directories that are writing for the row
offsets = [0, 10] if args.mtd and len(args.mtd.split(",")) == 2 else [0]



#Selects the columns that we are interested in, including combinations of some of the output columns (e.g. difference
#between 2 of them for 'overall', chosing one column or the other for 'ppcs', removing percentage signs, etc.

#For each directory that is producing results for the given file's row (defaults to just 1 directory)...
for i, row in model_preds.iterrows():
    #For each '--best' or '--worst' optional arguments set
    for j, bwm in enumerate(best_worst_metrics):
        #If it's set by the appropriate arg...
        if bwm:
            sel_row = [row["Short file name"], row["Source dir"], row["Model trained dir(s)"], row["Measurements tested"]]
            #For each of the model directories contained within the row...
            for offset in offsets:
                if bwm == "pacp":
                    #Adds the percentage value w/o the percentage symbol
                    sel_row.append(float(row[offset + 8][:-1]))
                elif bwm == "ppcs":
                    #Selects the correct column value to append to the list based on the value in 'True 'D/HC Label''
                    ppcs = float(row[offset + 11][:-1]) if row[offset + 9] == "D" else float(row[offset + 12][:-1])
                    sel_row.append(ppcs)
                else:
                    #Finds the absolute difference between the 'True 'Overall NSAA Score'' and 'Predicted 'Overall NSAA Score''
                    overall_diff = abs(int(row[offset + 13]) - int(row[offset + 14]))
                    sel_row.append(overall_diff)
            #Add the row w/ columns removed + redesigned to list of rows that we're interested in
            selected_rows[j].append(sel_row)


#Dictionary to map metric selected for given '--best' or '--worst' to a column name
metric_dict = {"pacp": "Percentage of acts correctly predicted", "ppcs": "Percentage of predicted correct sequences",
               "overall": "Diff true/pred overall NSAA score"}
#Sets up the column names for the 5 or 6 columns needed, based on the metrics stored in them and the names of the
#alt dirs or (if not set), the source dir, for each of the '--best' and '--worst' optional args set
col_names = [[], []]
for i in range(len(col_names)):
    if selected_rows[i]:
        col_names[i] += ["Short file name", "Source dir", "Model trained dir(s)", "Measurements tested"]
        for j in range(len(offsets)):
            met_col = "(" + args.mtd.split(",")[j] + ") : " + metric_dict[best_worst_metrics[i]] if args.mtd else \
                "(" + args.sd + ") : " + metric_dict[best_worst_metrics[i]]
            col_names[i].append(met_col)



#Finally, selects the top or bottom (or both) 'n' number of lines, depending on which of '--best' or '--worst' has been
#selected and the number of lines to extract from each of them, having reversed them if needed for percentage metrics
#('pacp' and 'ppcs'), before printing out the selected rows to the console as a DataFrame.


for i in range(len(selected_rows)):
    if selected_rows[i]:
        #Create a DataFrame of each group of rows with the previously-obtained column names
        df = pd.DataFrame(selected_rows[i], columns=col_names[i])
        #The rows are now sorted in either ascending or descending order (based on the metrics recorded in the rows),
        #with priority given to first ordering by the 5th column and secondary ordering importance to the 6th column
        if best_worst_metrics[i] == "pacp" or best_worst_metrics[i] == "ppcs":
            df = df.sort_values(by=[col_names[i][4], col_names[i][5]], ascending=False)
        else:
            df = df.sort_values(by=[col_names[i][4], col_names[i][5]], ascending=True)

        #Prints either the top or bottom 'best_worst_rows[i]' rows of the DataFrame, depending on the if it's printing
        #for '--best' or '--worst'
        if i == 0:
            print("\nBest performing " + str(best_worst_rows[i]) + "rows by " + col_names[i][4] +
                  " and " + col_names[i][5] + "...\n")
            df = df.iloc[:best_worst_rows[i], :]
            print(df)

        else:
            print("\nWorst performing " + str(best_worst_rows[i]) + "rows by " + col_names[i][4] +
                  " and " + col_names[i][5] + "...\n")
            df = df.iloc[(best_worst_rows[i]*-1):, :]
            #Reverses the order so the worst rows appear first
            df = df.iloc[::-1]
            print(df)



