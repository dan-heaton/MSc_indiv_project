---------------------		model_predictor.py Overview and explanation		---------------------


------ Motivation ------

While gaining insights into various types of model parameters, source data types, data preprocessing 
options, and so on are an important and useful output of the project, one of the primary aims is to be able 
to test models that have already been built on complete files; for example, we may wish to see how the model 
performs when tested with a subject it has never seen before and record the results in 'model_predictions.csv'. 
Alternatively, we may want to be using models in their 'production' form to help inform specialists about 
subjects based just on model results. To do this, we need a separate script that both preprocesses a single 
subject's file(s) for testing, but also to load the relevant models from a specified source.

The 'model_predictor.py' script was written with this in mind. While it may work with predictions and the 
preprocessing of data, it's unlike the 'rnn.py' script in that it does not create any models; rather, it 
uses the models that have been created by 'rnn.py' already. Hence, the script is only useable after 'rnn.py' 
has created the required models. The arguments to 'model_predictor.py' primarily serve three purposes: to 
load the data from the relevant source directories based on the file types (e.g. AD and jointAngle) in '.csv' 
format (created by either the 'comp_stat_vals.py' and 'ft_sel_red.py' scripts or the 'ext_raw_measures.py' 
script), to load the models that have been created that have been trained on the directory the file in 
question is sourced from and with the relevant file types and for all output types, and finally to run the 
'.csv' data files on the models that have been loaded and aggregage the results to make overall predictions.





------ How it works ------

The execution of 'model_predictor.py' runs in a fairly procedural manner; hence, it's more intuitive to 
describe the program as a sequence of steps that call functions when necessary rather than a series of 
functions that are connected together as needed (e.g. 'comp_stat_vals.py'). The execution is as follows:

1.) Checks the validity of each passed in argument.

2.) For a given file name, loads in the '.csv' files for each of the file types provided; for example, 
if fn='D4' and ft='AD,jointAngle,sensorMagneticField', then the 'FR_AD_D4_stat_features.csv', 
'D4_jointAngle.csv' and 'D4_sensorMagneticField.csv' files are loaded in (the names of which might slightly 
vary in practice due to naming conventions).

3.) Identify the directories that contain the models that we require to use for the files' assessment; note 
that these are all contained within the 'output_files\rnn_models' directory and have names that reflect 
how the models were built and on what data. This is done for all three output types as well. For example, 
if dir='NSAA' and ft='AD', then 'NSAA_AD_all_dhc_--seq_len=10_--seq_overlap=0.9_--epochs=300', 
'NSAA_AD_all_acts_--seq_len=10_--seq_overlap=0.9_--epochs=300' and 'NSAA_position_all_dhc_--seq_len=600_
--seq_overlap=0.9_--discard_prop=0.9' are loaded as the directory names containing the models.

4.) Preprocesses the data from the '.csv' files so that they will fit into the pre-trained models (e.g. by 
having the expected batch size and sequence length) along with fetching the requisite 'y labels' for the 
data in the same way as is done for 'rnn.py'.

5.) For each output type and for each of the '.csv' files of the data for the subject in question, 
put all the data through the model that corresponds to the '.csv's file type (e.g. jointAngle or AD) and 
its output type in prediction mode and have the predictions collected.

6.) For a given output type, average together all predictions made over every sequence prediction for every 
file type to get a prediction for that output type for the whole file. For example, for the NSAA overall 
score output type, we average the scores for every sequence from a given file type's predictions, repeat 
this for the other file types, and finally average these scores to get a prediction of the overall score 
that takes into account all predictions made for every sequence of all the file types we are assessing on.

7.) Outputs these scores to the user and appends these results to a new line within the 'model_predictions.csv' 
file, along with the name of the subject in question as well as the file types used, the source directory, etc.




Special attention should be paid to some of the optional arguments. Some are used exclusively by other 
calling scripts (e.g. '--handle_dash' and '--file_num' are exclusively used by the 'test_altdirs.py' script) 
and others are fairly simple and self-explanatory (e.g. '--show_graph' shows the true and predicted overall 
NSAA scores made for the subject, while '--single_act' is used when the input to the models are single-act 
files), there are a few others that require brief explanation:

-- '--alt_dirs': provide this with a name of a directory that is not the same as 'dir' to test files on 
models that haven't been trained on the same directory; for example, if dir='allmatfiles' and alt_dirs='NSAA' 
then subject files will be loaded from the 'allmatfiles' directory but tested on models trained on files 
originating from the 'NSAA' directory. The motivation and results of this are explored in more depth in 
the results discussion.

-- '--use_seen': for a given file name (e.g. matching or deriving from a subject short name like 'D2-009'), 
the default behaviour of the script is to seek out model directories where the subject has been completely 
left out of the training and testing process; in other words, the subject who we're assessing is completely 
new to the models assessing it. This is done by specifically seeking model directories with names containing 
'--leave_out=<file name>' (along with the other required directory and file type arguments). Sometimes, we 
may not want to do this specifically: for example, when we want to compare a subject being tested on a model 
familiar with the subject with one that isn't. Further results of this are explored in the 
experiments discussion.

-- '--use_balanced': in a similar way that '--use_seen' seeks out model directories that haven't got 
something in their names, this optional argument specifically seeks model directories to use that have 
got '--balanced=<up/down>' in the name (depending on the value given to '--use_balanced'). Hence, this 
allows us to test complete files on models that have trained on an up- or down-sampled data set. For more 
information on the data balancing process, consult the README for 'data_balancer.py', or for more info on 
how well this performed on complete files, seek the relevant section in experiment results.


