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
args = parser.parse_args()


#Note: CHANGE THESE to location of the 3 sub-directories' encompassing directory local to the user that's needed to
#map to the .csvs containing the NSAA information
local_dir = "C:\\msc_project_files\\"

#Other locations and sub-dir nameswithin 'local_dir' that will contain the files we need, as dictated by assuming the
#user has previously run the required scripts (e.g. 'comp_stat_vals', 'ext_raw_measures', etc.)
source_dir = local_dir + "output_files\\"
sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\"]
sub_sub_dirs = ["AD\\", "JA\\", "DC\\"]
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
    if args.single_act:
        if args.dir == "NSAA":
            source_dir += "act_files\\"
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
        full_file_names = [source_dir + match_fns[i] for i in range(len(match_fns))]
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



def ft_red_select(full_file_name):
    """
    :param 'full_file_name', which is the full path name to the file in question that we wish to do dimensionality
    reduction on
    :return: the new reduced 'x' and 'y' components of the file to be later written to a new file
    """
    print("Reducing dims of " + full_file_name + "...")
    df = pd.read_csv(full_file_name)
    col_names = df.columns.values[2:]
    #Splits the loaded file into the 'y' parts (the original .mat source file column and file label) and 'x' parts (all
    #the statistical values extracted via 'comp_stat_vals.py')
    x = df.iloc[:, 2:].values
    y = df.iloc[:, :2].values

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
    elif choice == "grp":
        grp = GaussianRandomProjection(n_components=num_components)
        new_x = grp.fit_transform(x)
    elif choice == "agglom":
        #Find out
        agg = FeatureAgglomeration(n_clusters=num_components)
        new_x = agg.fit_transform(x)
    elif choice == "thresh":
        #Below threshold gives ~26 components upon application
        vt = VarianceThreshold(threshold=0.00015)
        new_x = vt.fit_transform(x)
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

    #For each of the 3 tables of data that we have on the subjects, load in the table, find the columns with ID and
    #overall NSAA scores, and create a dictionary of matching values, e.g. {'D4': 15, 'D11: 28,...}, with all values
    #from each table
    mw_tab = pd.read_excel(local_dir + "6MW-matFiles\\6mw_matfiles.xlsx")
    mw_cols = mw_tab[["ID", "NSAA"]]
    mw_dict = dict(pd.Series(mw_cols.NSAA.values, index=mw_cols.ID).to_dict())

    nsaa_matfiles_tab = pd.read_excel(local_dir + "NSAA\\matfiles\\nsaa_matfiles.xlsx")
    nsaa_matfiles_cols = nsaa_matfiles_tab[["ID", "NSAA"]]
    nsaa_matfiles_dict = dict(pd.Series(nsaa_matfiles_cols.NSAA.values, index=nsaa_matfiles_cols.ID).to_dict())
    mw_dict.update(nsaa_matfiles_dict)

    kinedmd_tab = pd.read_excel(local_dir + "NSAA\\KineDMD data updates Feb 2019.xlsx")
    kinedmd_cols = pd.concat([kinedmd_tab.iloc[2:20, 0], kinedmd_tab.iloc[2:20, 70]], axis=1)
    kinedmd_cols.columns = ["ID", "NSAA"]
    kinedmd_dict = dict(pd.Series(kinedmd_cols.NSAA.values, index=kinedmd_cols.ID).to_dict())
    mw_dict.update(kinedmd_dict)

    #Adds column of overall NSAA scores at position 0 of every row of the data values, with the NSAA score being
    #appended determined by the short file name of the data as found at the beginning of each row of the data
    nss = [mw_dict[i] for i in [j.split("_")[0] for j in file_df.iloc[:, 0].values]]
    file_df.insert(loc=0, column="NSS", value=nss)

    #Loads the data that contains information about single act NSAA scores from the relevant .xlsx file, extracts the
    #file names and single-acts columns, and creates a list of label names (i.e. the names of the activities) and a
    #dictionary that maps the label names to a list of single-act scores
    nsaa_acts_tab = pd.read_excel(local_dir + "NSAA\\KineDMD data updates Feb 2019.xlsx")
    nsaa_acts_file_names = nsaa_acts_tab.iloc[2:20, 0].values
    nsaa_acts = nsaa_acts_tab.iloc[2:20, 53:70].values
    nsaa_acts_dict = dict(zip(nsaa_acts_file_names, nsaa_acts))
    nsaa_labels = nsaa_acts_tab.iloc[1, 53:70].values

    #For each label name and for every row, adds the score that is found in the single-acts dictionary for the relevant
    #activity for a given short file name (if it isn't found in the dictionary, add a '2' as we're assuming it's a
    #healthy control patient), add these together, and insert each new row of values at the beginning of the old rows
    #so each now have the additional single-act scores and overall NSAA scores at the beginning of each row and return it
    label_sample_map = []
    for i in range(len(nsaa_labels)):
        inner = []
        for j in range(len(file_df.index)):
            fn = file_df.iloc[j, 1].split("_")[0]
            if fn in nsaa_acts_dict:
                inner.append(nsaa_acts_dict[fn][i])
            else:
                #If patient isn't found in the 'KineDMB' table, assume its a healthy control patient and thus all
                #scores for all activities are perfect (i.e. '2').
                inner.append(2)
        label_sample_map.append(inner)
    for i in range(len(nsaa_labels)):
        file_df.insert(loc=(i+1), column=nsaa_labels[i], value=label_sample_map[i])
    return file_df



#For each of the full file names of the files that we wish to reduce the dimensions of, get the new 'x' and 'y'
#components of their reduced form, concatenate them together, add the overall and single-act NSAA scores if possible
#to do so (otherwise, don't continue with this file), and write this to the same directory it was sourced from as a
#.csv file with the same name except with a 'FR_' at the front of the file name
for full_file_name in full_file_names:
    new_x, y = ft_red_select(full_file_name)

    #Recombine the now-reduced 'x' data with the source file name and label columns
    new_df = pd.DataFrame(np.concatenate((y, new_x), axis=1))

    #Add a column of NSAA scores to the DataFrame by referencing the external .csvs
    try:
        new_df_nsaa = add_nsaa_scores(new_df)
    except KeyError:
        print(full_file_name + " not found as entry in either '6mw_matfiles.xlsx', 'nsaa_matfiles.xlsx', "
                               "or 'KineDMD data updates Feb 2019.xlsx', 'skipping...")
        continue

    #Writes the new data to the same directory as before with the same name except with 'FR_' on the front
    split_full_file_name = full_file_name.split("\\")
    split_full_file_name[-1] = "FR_" + split_full_file_name[-1]
    new_full_file_name = "\\".join(split_full_file_name)
    new_df_nsaa.to_csv(new_full_file_name)