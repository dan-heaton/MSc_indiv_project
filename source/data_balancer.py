from collections import OrderedDict
import pandas as pd
from random import shuffle
from random import randint


def ext_label_dist(file_name):
    """
        :param the name of the file name (not inc directory path) from which we wish to get the overall NSAA score
        :return the 'overall NSAA score' that corresponds to the file name
    """
    #Gets the short file name (e.g. 'D4') from the fill name
    if file_name.startswith("FR"):
        short_file_name = file_name.split("_")[2]
    elif file_name.startswith("AD"):
        short_file_name = file_name.split("_")[1]
    else:
        short_file_name = file_name.split("-")[0]

    #Loads the table of short-file-names-to-overall-NSAA-score, gets the relevant columns, and creates a dictionary
    #from these columns
    nsaa_6mw_tab = pd.read_excel("..\\documentation\\nsaa_6mw_info.xlsx")
    nsaa_6mw_cols = nsaa_6mw_tab[["ID", "NSAA"]]
    nsaa_overall_dict = dict(pd.Series(nsaa_6mw_cols.NSAA.values, index=nsaa_6mw_cols.ID).to_dict())

    #Ensures that short file names ending with 'V2' get a label (e.g. 'D6V2' uses the score for for 'D6') and, with
    #this, gets the overall NSAA score from the dictionary for the given short file name
    if short_file_name in nsaa_overall_dict:
        y_label_balance = nsaa_overall_dict[short_file_name]
    else:
        y_label_balance = nsaa_overall_dict[short_file_name[:-2]]
    return y_label_balance



def downsample(x_data, y_data, y_data_balance):
    """
        :param 'x_data' (the 'x' component of the data that we wish to balance), 'y_data' (the 'y' component of the
        data that we wish to balance), and 'y_data_balance' (corresponding overall NSAA scores for each sample of 'x'
        and 'y' that are used to determine which x and y samples to remove or keep; not this is the same as 'y_data' if
        'choice' arg from 'rnn.py' is 'overall')
        :return the downsampled x and y components of data as 'new_x' and 'new_y', with strings to print later on in
        'rnn.py' (at the end of training and testing) returned here as 'out_strs'
    """

    out_strs = []
    #Creates a dictionary of overall NSAA scores and their frequency within 'x' and 'y' components in ascending
    #frequency order
    label_dict = {l: y_data_balance.count(l) for l in set(y_data_balance)}
    label_count = sorted(label_dict.items(), key=lambda x: x[1])
    out_strs.append("Overall NSAA labels for sequences and freq in Y before downsampling: " +
                    str(dict(OrderedDict(sorted(label_dict.items(), key=lambda t: t[0])))))
    #Shuffles the zipped 'x' and 'y' data so they both get shuffled in the same way so the samples that we remove are
    #completely randomised within the sample
    Z = list(zip(x_data, y_data, y_data_balance))
    shuffle(Z)
    x_data, y_data, y_data_balance = zip(*Z)

    #Gets the lowest frequency of a given overall NSAA score; this is the frequency of each overall NSAA scores'
    #corresponding 'x' and 'y' samples should be reduced to
    tar_len = label_count[0][1]
    dict_count = {}
    new_x, new_y = [], []

    #For each zipped x sample, y sample, and corresponding overall NSAA score, add 1 to their corresponding overall NSAA
    #score in the 'dict_count' dictionary and, if it's below the lowest frequency of scores, add the x and y samples to
    #their respective lists; this ensures that, for a given overall NSAA score, there are never more than 'tar_len'
    #x and y samples added to the new data lists
    for x, y, z in zip(x_data, y_data, y_data_balance):
        if z in dict_count:
            if dict_count[z] < tar_len:
                dict_count[z] += 1
                new_x.append(x)
                new_y.append(y)
        else:
            dict_count[z] = 1
            new_x.append(x)
            new_y.append(y)

    out_strs.append("Overall NSAA labels for sequences and freq in Y after downsampling: " +
                    str(dict(OrderedDict(sorted(dict_count.items(), key=lambda t: t[0])))))
    return new_x, new_y, out_strs



def upsample(x_data, y_data, y_data_balance):
    """
        :param 'x_data' (the 'x' component of the data that we wish to balance), 'y_data' (the 'y' component of the
        data that we wish to balance), and 'y_data_balance' (corresponding overall NSAA scores for each sample of 'x'
        and 'y' that are used to determine which x and y samples to remove or keep; not this is the same as 'y_data' if
        'choice' arg from 'rnn.py' is 'overall')
        :return the upsampled x and y components of data as 'new_x' and 'new_y', with strings to print later on in
        'rnn.py' (at the end of training and testing) returned here as 'out_strs'
    """
    out_strs = []
    #Creates a dictionary of overall NSAA scores and their frequency within 'x' and 'y' components in ascending
    #frequency order
    label_dict = {l: y_data_balance.count(l) for l in set(y_data_balance)}
    label_count = sorted(label_dict.items(), key=lambda x: x[1])
    out_strs.append("Overall NSAA labels for sequences and freq in Y before upsampling: " +
                    str(dict(OrderedDict(sorted(label_dict.items(), key=lambda t: t[0])))))

    #Gets the highest frequency of a given overall NSAA score; this is the frequency of each overall NSAA scores'
    #corresponding 'x' and 'y' samples should be increased to by random sampling
    tar_len = label_count[-1][1]

    #Dictionary contains keys as 'y_data_balance' values (i.e. overall NSAA values), with values being a list of
    #2-element lists, with 1st elem being an x-val and 2nd being a y-val, each pair corresponding to a value
    #in y_data_balance
    class_dict = {}
    for x, y, z in zip(x_data, y_data, y_data_balance):
        if z in class_dict:
            class_dict[z].append([x, y])
        else:
            class_dict[z] = [[x, y]]

    dict_count = {k: 0 for k in class_dict}
    new_x, new_y = [], []
    #For each overall NSAA score, we sample from it's list of x/y sample pairs at a random point within the entirety
    #of it's list length ('rand_pos') and add each part to their respective 'new_x' and 'new_y' lists (note that as we
    #are not removing it from the original list when adding to 'new_x', this is sampling with replacement)
    for key in class_dict:
        for i in range(tar_len):
            rand_pos = randint(0, len(class_dict[key])-1)
            new_x.append(class_dict[key][rand_pos][0])
            new_y.append(class_dict[key][rand_pos][1])
            dict_count[key] += 1

    out_strs.append("Overall NSAA labels for sequences and freq in Y after downsampling: " +
                    str(dict(OrderedDict(sorted(dict_count.items(), key=lambda t: t[0])))))

    return new_x, new_y, out_strs