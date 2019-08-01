import pandas as pd
import numpy as np
import os
import sys
import argparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from settings import local_dir, source_dir, sub_dirs, sub_sub_dirs


"""Section below encompasses all the arguments that are required to setup and train the model. This includes the name 
of the directory to use to load the file(s) to reduce the dimensions of, the type of file that we are expected to 
be dealing with, the short name of the file to reduce the dimensions of", and the choice of feature selection/reduction 
technique to use to process the file. The script also normalizes the data by default, so specify '--no_normalize' if 
this isn't desired, along with '--dis_kept_features' to print the features selected and '--num_features' to display 
the number of features chosen by the script."""
parser = argparse.ArgumentParser()
parser.add_argument("dir", help="Specifies which source directory to use so as to process the files contained within "
                                "them accordingly. Must be one of '6minwalk-matfiles', '6MW-matFiles' or 'NSAA'.")
parser.add_argument("ft", help="Specify type of .mat file that the .csv is to come from, being one of 'JA' (joint "
                               "angle), 'AD' (all data), or 'DC' (data cube).")
parser.add_argument("fn", help="Specify the short file name of a .csv to load from 'source_dir'; e.g. for file "
                               "'All_D2_stats_features.csv', enter 'D2'. Specify 'all' for all the files available "
                               "in the 'source_dir'.")
parser.add_argument("choice", help="Specifies the choice of feature reduction or feature selection to carry out on "
                                   "the input .csv data")
parser.add_argument("--no_normalize", type=bool, nargs="?", const=True,
                    help="Include this argument if user specifically doesn't want to normalize the 'x' data.")
parser.add_argument("--num_features", type=int,
                    help="Specify an 'int' here to define the features to reduce the data to.")
parser.add_argument("--dis_kept_features", type=bool, nargs="?", const=True, default=False,
                    help="Specify this if user wishes to print to console the kept features after a feature selection "
                         "choice is made. Has no impact if an unsupervised feature reduction choice is made (e.g. "
                         "'pca', 'grp', or 'agglom').")
parser.add_argument("--combine_files", type=bool, nargs="?", const=True, default=False,
                    help="Specify this to concatenate all the files before reducing them, to then reduce the features "
                         "on this combined set before separating them to write them to output files as standard. "
                         "Ensures that all files are reduced in the same way.")
args = parser.parse_args()

choices = ["pca", "grp", "agglom", "thresh", "rf"]

#Default number of components to preserve with the feature selection/reduction options; overridden if
#user supplies value for '--num_features'
num_components = 30

#Appends the sub_dir name to 'source_dir' if it's one of the allowed names
if args.dir + "\\" in sub_dirs:
    source_dir += args.dir + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'direct_csv'.")
    sys.exit()

#Appends the sub_sub_dir name to 'source_dir' if it's an allowed name, along with appending 'act_files\\' if it's 'NSAA'
if args.ft.upper() + "\\" in sub_sub_dirs:
    source_dir += args.ft + "\\"
else:
    print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
          "\'JA', or \'DC\'.")
    sys.exit()


#Find the matching full file name in 'source_dir' given the 'fn' argument, otherwise use all file names in 'source_dir'
#if 'fn' is 'all'
match_fns = [s for s in os.listdir(source_dir) if s.split(".")[0].split("_")[1].upper() == args.fn.upper()]
if match_fns:
    full_file_names = [source_dir + match_fns[0]]
else:
    if args.fn == "all":
        match_fns = [s for s in os.listdir(source_dir)]
        full_file_names = [source_dir + match_fns[i] for i in range(len(match_fns)) if "FR_" not in match_fns[i]
                           and match_fns[i].endswith(".csv")]
    else:
        print("Third arg ('fn') must be the short name of a file (e.g. 'D2' or 'all') within", source_dir)
        sys.exit()

#Sets 'choice' equal to the 'choice' argument if it's one of the allowed feature selection/reduction techniques
#(e.g. pca, thresh, rf, etc.)
if args.choice in choices:
    choice = args.choice
else:
    print("Fourth arg ('choice') must be the name a feature selection/reduction and must be one of 'pca' (to carry "
          "out PCA on the file), 'grp' (to reduce dimensionality through a Gaussian random projection), 'agglom' "
          "(to carry out feature agglomeration), 'thresh' (to do features selection based on a minimal variance "
          "threshold for each features), and 'rf' (to fit the data to a random forest and use this to select "
          "the most useful features).")
    sys.exit()



def ft_red_select(x, y):
    """
    :param 'full_file_name', which is the full path name to the file in question that we wish to do dimensionality
    reduction on
    :return: the new reduced 'x' and 'y' components of the file to be later written to a new file
    """

    #Normalize the data
    if not args.no_normalize:
        x = normalize(x)

    #Overrides the number of features to keep in feature selection/reduction if optional argument is chosen
    global num_components
    if args.num_features:
        num_components = args.num_features

    #Given the argument choice of feature selection/reduction, creates the relevant object, fits the 'x' data to it,
    #and reduces/transforms it to a lower dimensionality
    new_x = []
    print("Original 'x' shape:", np.shape(x))
    if choice == "pca":
        pca = PCA(n_components=num_components)
        new_x = pca.fit_transform(x)
        print("Explained variance = " + str(round(sum(pca.explained_variance_)*100, 2)) + "%" )
    elif choice == "grp":
        grp = GaussianRandomProjection(n_components=num_components)
        new_x = grp.fit_transform(x)
    elif choice == "agglom":
        agg = FeatureAgglomeration(n_clusters=num_components)
        new_x = agg.fit_transform(x)
    elif choice == "thresh":
        #Below threshold gives ~26 components upon application
        vt = VarianceThreshold(threshold=0.00015)
        new_x = vt.fit_transform(x)
        print("Explained variance = " + str(round(sum(vt.variances_)*100, 2)) + "%")
        kept_features = list(vt.get_support(indices=True))
        if args.dis_kept_features:
            print("Kept features: ")
            for i in kept_features:
                print(col_names[i])
    elif choice == "rf":
        y_labels = [1 if s == "D" else 0 for s in y[:, 1]]
        clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        print("Fitting RF model....")
        clf.fit(x, y_labels)
        sfm = SelectFromModel(clf, threshold=-np.inf, max_features=num_components)
        print("Selecting best features from model...")
        sfm.fit(x, y_labels)
        kept_features = list(sfm.get_support(indices=True))
        if args.dis_kept_features:
            print("Kept features: ")
            for i in kept_features:
                print(col_names[i])
        new_x = x[:, kept_features]

    print("Reduced 'x' shape:", np.shape(new_x))
    return new_x, y



def add_nsaa_scores(file_df):
    """
    :param 'file_df', which contains the values in a 2D numpy array, to have the values NSAA scores appended on each
    of its rows
    :return: the same data as before, but with the overall and individual NSAA scores appended at the beginning of
    each row of the data
    """
    #To make sure that accepted parameter is as a DataFrame
    file_df = pd.DataFrame(file_df)

    #For the table of data that we have on the subjects, load in the table, find the columns with ID and
    #overall NSAA scores, and create a dictionary of matching values, e.g. {'D4': 15, 'D11: 28,...}, with all values
    #from each table
    nsaa_6mw_tab = pd.read_excel("..\\documentation\\nsaa_6mw_info.xlsx")
    nsaa_6mw_cols = nsaa_6mw_tab[["ID", "NSAA"]]
    nsaa_overall_dict = dict(pd.Series(nsaa_6mw_cols.NSAA.values, index=nsaa_6mw_cols.ID).to_dict())

    # If the first column's names begins with "jointangle", remove this part
    if file_df.iloc[0, 0].startswith("jointangle"):
        for i, row in file_df.iterrows():
            row[0] = row[0].split("jointangle")[1]

    #Adds column of overall NSAA scores at position 0 of every row of the data values, with the NSAA score being
    #appended determined by the short file name of the data as found at the beginning of each row of the data
    nss = [nsaa_overall_dict[i.upper()[:-2] if i.upper().endswith("V2") else i.upper()]
           for i in [j.split("_")[0] for j in file_df.iloc[:, 0].values]]
    file_df.insert(loc=0, column="NSS", value=nss)

    #Loads the data that contains information about single act NSAA scores from the .xlsx file, extracts the
    #file names and single-acts columns, and creates a list of label names (i.e. the names of the activities) and a
    #dictionary that maps the label names to a list of single-act scores
    nsaa_single_dict = {}
    for name, acts in zip(nsaa_6mw_tab.loc[:, "ID"].values, nsaa_6mw_tab.iloc[:, 5:].values):
        if not any(np.isnan(acts)):
            nsaa_single_dict[name] = acts
    nsaa_act_labels = nsaa_6mw_tab.columns.values[5:]

    #For each label name and for every row, adds the score that is found in the single-acts dictionary for the relevant
    #activity for a given short file name (if it isn't found in the dictionary, add a '2' as we're assuming it's a
    #healthy control patient), add these together, and insert each new row of values at the beginning of the old rows
    #so each now have the additional single-act scores and overall NSAA scores at the beginning of each row and return it
    label_sample_map = []
    for i in range(len(nsaa_act_labels)):
        inner = []
        for j in range(len(file_df.index)):
            fn = file_df.iloc[j, 1].split("_")[0].upper()
            fn = fn[:-2] if fn.endswith("V2") else fn
            if fn in nsaa_single_dict:
                inner.append(nsaa_single_dict[fn][i])
            elif fn.startswith("HC"):
                inner.append(2)
            else:
                # If patient isn't found in the table (and thus we don't have info on the individual NSAA scores),
                # don't continue with the file and move onto the next one
                raise KeyError
        label_sample_map.append(inner)
    for i in range(len(nsaa_act_labels)):
        file_df.insert(loc=(i + 1), column=nsaa_act_labels[i], value=label_sample_map[i])
    return file_df



#Sets the scope of variables set within the 'if args.combine_files:' statement
x_combine_files, y_combine_files = None, None
empty_arrays = True
file_lens = []

#For each of the full file names of the files that we wish to reduce the dimensions of, get the new 'x' and 'y'
#components of their reduced form, concatenate them together, add the overall and single-act NSAA scores if possible
#to do so (otherwise, don't continue with this file), and write this to the same directory it was sourced from as a
#.csv file with the same name except with a 'FR_' at the front of the file name. If the 'args.commbine_files' argument
#was set, however, just concatenate all the data along axis 0 to prepare for further processing

for full_file_name in full_file_names:
    print("Extracting data of " + full_file_name + "...")
    df = pd.read_csv(full_file_name)
    col_names = df.columns.values[2:]
    # Splits the loaded file into the 'y' parts (the original .mat source file column and file label) and 'x' parts (all
    # the statistical values extracted via 'comp_stat_vals.py')
    x = df.iloc[:, 2:].values
    y = df.iloc[:, :2].values
    if not args.combine_files:
        print("Reducing dims of " + full_file_name + "...")
        try:
            new_x, new_y = ft_red_select(x, y)
        except ValueError:
            print("'" + full_file_name + "': number of rows too few for number of features, skipping...")
            continue
        #Recombine the now-reduced 'x' data with the source file name and label columns
        new_df = pd.DataFrame(np.concatenate((new_y, new_x), axis=1))

        #Add a column of NSAA scores to the DataFrame by referencing the external .csvs
        try:
            new_df_nsaa = add_nsaa_scores(new_df)
        except KeyError:
            print(full_file_name + " not found as entry in either 'nsaa_6mw_info', skipping...")
            continue

        #Writes the new data to the same directory as before with the same name except with 'FR_' on the front
        split_full_file_name = full_file_name.split("\\")
        split_full_file_name[-1] = "FR_" + split_full_file_name[-1]
        new_full_file_name = "\\".join(split_full_file_name)
        if os.path.exists(new_full_file_name):
            os.remove(new_full_file_name)
        new_df_nsaa.to_csv(new_full_file_name)
    #Creates a new data storage table if it doesn't exist; otherwise, appends to existing table, along with adding the
    #length of the file to the list of file lengths used to separate the file data after dimensionality reduction
    else:
        file_lens.append(len(x))
        if empty_arrays:
            x_combine_files = x
            y_combine_files = y
            empty_arrays = False
        else:
            x_combine_files = np.append(x_combine_files, x, axis=0)
            y_combine_files = np.append(y_combine_files, y, axis=0)


#If we're combining the files for dimensionality reduction, take the concatenated data table of all the files, reduce
#the dimensions using the chosen technique and, for each of the source file names and their corresponding file lengths,
#extract the first 'file_len' rows from the table, remove these rows from the table, and write these rows to a new
#file name with 'FRC_' appended at the front
if args.combine_files:
    print("\nReducing dims of all files...\n")
    x_total, y_total = ft_red_select(x_combine_files, y_combine_files)
    for full_file_name, file_len in zip(full_file_names, file_lens):
        new_x, new_y = x_total[:file_len], y_total[:file_len]
        x_total, y_total = x_total[file_len:], y_total[file_len:]
        #Recombine the now-reduced 'x' data with the source file name and label columns
        new_df = pd.DataFrame(np.concatenate((new_y, new_x), axis=1))
        #Add a column of NSAA scores to the DataFrame by referencing the external .csvs
        try:
            new_df_nsaa = add_nsaa_scores(new_df)
        except KeyError:
            print(full_file_name + " not found as entry in either 'nsaa_6mw_info', skipping...")
            continue

        #Writes the new data to the same directory as before with the same name except with 'FRC_' on the front
        split_full_file_name = full_file_name.split("\\")
        split_full_file_name[-1] = "FRC_" + split_full_file_name[-1]
        new_full_file_name = "\\".join(split_full_file_name)
        if os.path.exists(new_full_file_name):
            os.remove(new_full_file_name)
        new_df_nsaa.to_csv(new_full_file_name)