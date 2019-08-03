---------------------		test_aldirs.py Overview and explanation		---------------------


------ Motivation ------

A key motivation of this project is investigating how well, if at all, models that are build on one type of data 
can be adapted to be used on other types of data. Furthermore, to get a good idea of how well this is done, it's 
necessary to test numerous files on pre-trained models. And in the case of testing natural movement files on 
models that are trained on NSAA and 6 minute walk files, this would require running 'model_predictor.py' manually 
over 400 times and each time with a different file name from within 'allmatfiles'. To get around this, 
'test_altdirs.py' was created to automate this process.

Crucially, this script only allows 'model_predictor.py' to work on assessing models' performances on unseen files 
that also have aren't trained on the same type of data. This allows us to see the strength of the correlation 
between different types of assessment for subjects wearing the suits and also whether or not predicting the 
assessment scores by models trained on one type of assessment can be used to infer assessment scores of data in 
a form that the models haven't been trained on. In other words, can we have subjects just do natural movement 
activites and then use the models that have been trained on NSAA and 6 minute walk assessments to determine 
their D/HC classification, NSAA overall scores, etc. just as well as if they had instead done the NSAA and 
6 minute walk assessments? The results of this are explored later in the relevant results discussions.



------ How it works ------

The 'test_altdirs.py' does the following when run:

1.) Reads in the name of a directory from which we wish to source the files that we wish to use for assessment, 
and also the names of the directories that will have been used to train certain models (for example, supplying 
'NSAA_6minwalk-matfiles' here will ensure that each time 'model_predictor.py' is then called it retrieves the 
models that are trained on NSAA and 6 minute walk files).

2.) Retrieves a list of .mat file names from within the source directory (i.e. if 'allmatfiles' was passed as 
the 'dir' argument then the names of all .mat files from within 'allmatfiles' were retrieved and stored in a list)

3.) For every file name within this list of file names, create a unique string that corresponds to the input 
string to run the 'model_predictor.py' script with certain arguments. This string includes the short file name 
of the file in question, the file types that the models will have been trained on (for example, if 'allmatfiles' 
was chosen as 'dir', then this must be 'jointAngle' as this is presently the only type of information that can 
be extracted from this type of data), the assessment file directory, and the source file directories that 
were used to train the models.

4.) From here, all functionality is passed on to the 'model_predictor.py', which runs once for every file within 
the source directory as specified by the 'dir' argument. For further information on how this runs and what it 
produces, refer to 'README 'model.predictor.py''.