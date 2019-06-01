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

#Note: CHANGE THESE to location of the 3 sub-directories' encompassing directory local to the user that's needed to
#map to the .csvs containing the NSAA information
nsaa_table_path = "C:\\msc_project_files\\"

#Note: CHANGE THIS to location of the source data for the feature selection/reduction to use
source_dir = "output_files\\"

sub_dirs = ["6minwalk-matfiles\\", "6MW-matFiles\\", "NSAA\\", "direct_csv\\"]
sub_sub_dirs = ["AD\\", "JA\\", "DC\\"]
choices = ["pca", "grp", "agglom", "thresh", "rf"]

#Default number of components to preserve with the feature selection/reduction options; overridden if
#user supplies value for '--num_features'
num_components = 30

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

#TODO: add optional args to choose # dimensions to keep
#TODO: add option to print out the kept dimensions of feature selection techniques
args = parser.parse_args()

if args.dir + "\\" in sub_dirs:
    source_dir += args.dir + "\\"
else:
    print("First arg ('dir') must be a name of a subdirectory within source dir and must be one of "
          "'6minwalk-matfiles', '6MW-matFiles', 'NSAA', or 'direct_csv'.")
    sys.exit(1)

if args.ft.upper() + "\\" in sub_sub_dirs:
    source_dir += args.ft + "\\"
else:
    print("Second arg ('ft') must be a name of a sub-subdirectory within source dir and must be one of \'AD\',"
          "\'JA', or \'DC\'.")
    sys.exit(1)


match_fns = [s for s in os.listdir(source_dir) if s.split(".")[0].split("_")[1].upper() == args.fn.upper()]
if match_fns:
    full_file_name = source_dir + match_fns[0]
else:
    print("Third arg ('fn') must be the short name of a file (e.g. 'D2' or 'all') within", source_dir)
    sys.exit(1)

if args.choice in choices:
    choice = args.choice
else:
    print("Fourth arg ('choice') must be the name a feature selection/reduction and must be one of 'pca' (to carry "
          "out PCA on the file), 'grp' (to reduce dimensionality through a Gaussian random projection), 'agglom' "
          "(to carry out feature agglomeration), 'thresh' (to do features selection based on a minimal variance "
          "threshold for each features), and 'rf' (to fit the data to a random forest and use this to select "
          "the most useful features).")
    sys.exit(1)



df = pd.read_csv(full_file_name)
col_names = df.columns.values[2:]
#Splits the loaded file into the 'y' parts (the original .mat source file column and file label) and 'x' parts (all
#the statistical values extracted via 'matfiles_analysis.py')
x = df.iloc[:, 2:].values
y = df.iloc[:, :2].values

#Normalize the data
if not args.no_normalize:
    x = normalize(x)

#Overrides the number of features to keep in feature selection/reduction if optional argument is chosen
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


def add_nsaa_scores(file_df):
    #To make sure that accepted parameter is as a DataFrame
    file_df = pd.DataFrame(file_df)
    mw_tab = pd.read_excel(nsaa_table_path + "6MW-matFiles\\6mw_matfiles.xlsx")
    mw_cols = mw_tab[["ID", "NSAA"]]
    mw_dict = dict(pd.Series(mw_cols.NSAA.values, index=mw_cols.ID).to_dict())

    nsaa_matfiles_tab = pd.read_excel(nsaa_table_path + "NSAA\\matfiles\\nsaa_matfiles.xlsx")
    nsaa_matfiles_cols = nsaa_matfiles_tab[["ID", "NSAA"]]
    nsaa_matfiles_dict = dict(pd.Series(nsaa_matfiles_cols.NSAA.values, index=nsaa_matfiles_cols.ID).to_dict())

    mw_dict.update(nsaa_matfiles_dict)
    nss = [mw_dict[i] for i in [j.split("_")[0] for j in file_df.iloc[:, 0].values]]
    file_df.insert(loc=0, column="NSS", value=nss)

    nsaa_acts_tab = pd.read_excel(nsaa_table_path + "NSAA\\KineDMD data updates Feb 2019.xlsx")
    nsaa_acts_file_names = nsaa_acts_tab.iloc[2:20, 0].values
    nsaa_acts = nsaa_acts_tab.iloc[2:20, 53:70].values
    nsaa_acts_dict = dict(zip(nsaa_acts_file_names, nsaa_acts))
    nsaa_labels = nsaa_acts_tab.iloc[1, 53:70].values

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



#Recombine the now-reduced 'x' data with the source file name and label columns
new_df = pd.DataFrame(np.concatenate((y, new_x), axis=1))

#Add a column of NSAA scores to the DataFrame by referencing the external .csvs
new_df_nsaa = add_nsaa_scores(new_df)

#Writes the new data to the same directory as before with the same name except with 'FR_' on the front
split_full_file_name = full_file_name.split("\\")
split_full_file_name[-1] = "FR_" + split_full_file_name[-1]
new_full_file_name = "\\".join(split_full_file_name)
new_df_nsaa.to_csv(new_full_file_name)