# Recurrent Neural Networks and Conditional Random Fields for Activity Recognition in Human Movement
## Overview
* Ongoing MSc project applying RNNs and CRFs to human movement data of children with Duchenne’s
muscular dystrophy, working with Great Ormond Street Hospital as part of a wider initiative (see:
https://www.imperial.ac.uk/news/184530/ai-trial-help-accelerate-future-treatments/)

* Data provided shows bodysuit captures of subjects with DMD undergoing North Star Ambulatory Assessment
and walking trials to assess severity of their condition, with the aims of the project being to assist specialists
assessing new subject’s North Star scores and identify the most influential activities on subject’s assessment

* Project involves collaboration with senior department research associates, large amount of data preparation,
working with numerous different types of raw measurements from different sensors, statistical feature
extraction, data visualization, feature selection and reduction, and model training, tuning, and selection

* Implementing entirety of the project in Python and involves working with numerous input/output file types,
along with using TensorFlow extensively to implement, train, and load RNN models, using scikit-learn for
various machine learning tools and techniques, and using pandas and numpy for processing data

## Repo Layout

![Script pipeline diagram](https://raw.githubusercontent.com/dan-heaton/MSc_indiv_project/blob/master/plans_and_presentations/Report%20stuff/Script%20pipeline%20diagram.PNG)

The following provides a brief overview of how the repo is organized, the significance of the individual directories, etc.

* **background_research**: this folder contains some preparatory programs that were written (with help from sources cited in the scripts themselves) to familiarize oneself with the building of RNNs/CRFs with the chosen libraries in order to make using them for the actual project later that much easier; hence, these aren't used by the rest of the project at all and are included for historical documentation reasons only.

* **paper_reviews**: contains paper reviews done of research papers that are believed to be relevant to the project. These predominantly focus on the use of RNNs and CRFs when applied to real-world human movement data, and each paper consists of a slightly-shortened bullet pointed version of the paper and then a section of the most significant points from these bullet points. Hence, these papers are useful in justifying decisions taken with respect to model choices, experimentation directions, etc., and will also be heavily used in construction of the final project report.

* **plans_and_presentations**: contains a collection of presentations that have been created to display to group members about the project's progress thus far (which are kept in order to be used in final report writings at a later date), several initial reports and other documentations of project progress thus far, and also 'MSc Project Plan.ods', which is where the already-completed and upcoming task lists are stored; this is particularly useful if one wishes to see what is currently being worked on or has recently been completed.

* **project_files**: as the name suggests, this contains the bulk of the files that are relevant to the actual data pipeline and the results obtained from it thus far. Broadly speaking, it is divided into two subdirectories:
  * **documentation**: contains a number of files pertaining to the outputs of the data pipeline for the project. This includes 'RNN Results.xlsx', which covers the performances of various model setups on test data (i.e. a large proportion of the experimentation covering different types of raw measurements, sequence lengths, overlaps, etc., sources their results from here), 'model_predictions.csv', which (unlike 'RNN Results.xlsx') shows the performance of using 'model_predictor.py' to assess the performance of pretrained models on whole data files, 'model_shapes.xlsx', which is just to be used by 'model_predictor.py' to set the sequence length to the correct value (and is not particularly relevant to the user), and 'nsaa_6mw_info.xlsx', which contains a table of the subject names and their corresponding single-act and overall NSAA scores (this provides the necessary 'y-labels' for many RNN models).
  
    We also have several other subdirectories in here:
      * **Graphs**: all graphs created by 'graph_creator.py' are placed in here. These source from 'RNN Results.xlsx' and 'model_predictions.csv' to create graphs that are easier to display the results of groups of experiments done.
      * **Script Explanations**: collection of 'READMEs' for each of the scripts within 'project_files\source'. The idea is that, if one wishes to find out what each script does, why it was written, etc., then reading its relevant 'README' should provide sufficient detail.
* **source**: contains all of the scripts that are needed by the project pipeline to run. This includes the core Python scripts (such as 'comp_stat_vals.py' and 'rnn.py', along with some 'supporting' scripts, such as 'settings.py' (to contain global variables that are used across several scripts) and a few batch scripts to automate some of the running of the scripts. For further info about the significance of any or all of the scripts, consult the 'README(s)' for the relevant scripts in 'Script Explanations', the script ecosystem overview in 'plans_and_presentations', or the diagram of the scripts and their connections to each other found in 'Source'.