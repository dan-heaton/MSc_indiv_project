---------------------		predictions_selector.py Overview and explanation		---------------------


------ Motivation ------

With so many file predictions being made and stored in the 'model_predictions.csv', it's become necessary 
to have a way to sort through them all and return the files that we are most interested in. This is why this 
script has been built: to filter rows of the table (each corresponding to a complete file prediction made 
using 'model_predictor.py' or by extension the 'test_altdirs.py' script) based on several arguments (e.g. 
the subject names we're interested in, the directory the subject was trained on, or the alt directories that 
the models were trained on if they are 'altdirs' rows) and, based on whether '--best' or '--worst' is 
provided, return the best 'm' rows according to output metric 'n', where these are provided as part of 
'--best'/'--worst' (e.g. '--best=30,overall').

In essence, this functions similarly to how an SQL query would operate as 'SELECT <a> FROM model_predictions 
WHERE <condition>. However, the desire was to do this in Python so the whole pipeline would only require 
one language for implementation (no accounting for libraries built on top of languages like C++, e.g. for 
TensorFlow). Furthermore, this is easily possible via extensive use of the 'pandas' library to load in 
'model_predictions.csv' as a DataFrame object, which is excellent for the filtering of rows based on cell 
values, ordering rows by lowest/highest values in a specified column, and so on to make manipulation of the 
table as easy as using an SQL query. Additionally, this also means that anyone else running this system 
only needs to setup a single language/IDE in order to execute all of the scripts.

The idea from building this script is having an easy way to see some of the 'most relevant' rows of the 
table to the user. Presently, this just takes the form of console output, though easy modification to have 
these lines written to file is possible. This script is especially useful for when we have many files to 
'sift' through in order to get an idea of which are the best or worst performing on a given metric. For 
example, one particular application could be using the script to look at all the natural movement behaviour 
files that have been assessed on models build on NSAA and 6-minute walk files (totalling ~400 files) and 
selecting the best 20 of these according to which predicts the overall NSAA score of that file closest to 
the true value for that file. This has the potential to help us identify the types of natural behaviour 
files (e.g. sitting and eating, playing, sitting and moving on the floor, etc.) perform the best according 
to the metric. Another application could be, for a given subject name from the NSAA directory and on models 
trained on the same directory but left out of the training set completely, which options make the subject 
be predicted closest to the correct score (e.g. if the models data are upsampled, downsampled, trained on 
single-act files, etc.).






------ How it works ------

The script itself is fairly simple with no functions to call or classes to instantiate; rather, it executes 
a series of 'groups' of instructions that carries out the above-outlined tasks based on the script arguments. 
These can be summarised as follows:

1.) Loads in the 'model_predictions.csv' file as a DataFrame object.

2.) Filters the rows of the table based on the 'sfn' arg, which removes all rows where the subject name 
doesn't match the value of 'sfn'; alternatively, if 'sfn'=all, keep all rows at this point.

3.) Filters the rows of the table based on the 'sd' arg, which removes all rows whose source directory 
column is different from the value of 'sd'.

4.) If the 'mtd' is given (i.e. if we're concerned with 'altdir' rows), filters the rows of the table based 
on this arg, which removes all rows whose altdir column is different from the arg value. Note that the this 
arg is given as comma-separated values, which corresponds to the list values of the column in question.

5.) Based on whether the optional 'best' or 'worst' args are given (or both), extracts the first part of 
the arg(s) as the number of best/worst lines in the table and the second part as the short name of the 
metric to use to determine which are the best/worst (i.e. by deciding which of the output columns of the 
table to use to order the rows).

6.) For each of the remaining rows of the tables (i.e. after having been filtered by steps 1-4), now filter 
the columns of the table: the first four columns are kept (the subject name, source dir, model trained dirs, 
and measurements tested), followed by one of the output columns (the column in question is selected by the 
second part(s) of the best/worst args). These values are additionally preprocessed: e.g. if 'overall' is 
selected, then the absolute value of the difference between the true and predicted values in their 
respective columns are selected, while if we're using the 'percentage of predicted correct sequences' metric, 
the relevant column for 'percentage of predicted <D, HC> sequences' is used based on the true D/HC label 
for the row.

7.) Creates a list of column names to create a new table of the top 'n' results that include the 
aforementioned 4 beginning column names from 'model_predictions.csv', followed by column names of the 
output metrics with the names of the dir that the models that outputted this metric were trained using.

8.) Finally, select the top or bottom (or both) 'n' number of lines based on the selected column metric, 
depending on which of '--best' or '--worst' has been selected and the number of lines to extract from each 
of them, having reversed the if needed for percentage metrics ('pacp' and 'ppcs'), before printing out 
the selected rows to the console as a DataFrame object.





