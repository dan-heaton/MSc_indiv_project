
---------------------		file_renamer.py Overview and explanation		---------------------


------ Motivation ------

One of the primary problems with working with .mat files as part of this project is the lack of standardization 
of file names as they were collected. We have primarily been dealing with 4 source directories containing .mat 
files: 'NSAA' (containing NSAA assessments of subjects), '6minwalk-matfiles' and '6MW-matFiles' (containing the 
6 minute walk assessments of subjects), and 'allmatfiles' (containing the natural movement files of subjects 
wearing the suit). Each directory had its own primary way of labelling files but, at the same time, it wasn't 
necessarily consistent throughout the directory.

This posed a not-insignificant problem in that some of basic characteristics of the file were determined by 
its file name (e.g. whether it was a 'D' or 'HC' file came from reading its file name, along with what subject 
the file was associated with). Until development of this script, the solution was having multiple ways of 
processing every file name within the various scripts that need them. However, there's several flaws in 
this approach:

1.) It was not particularly extensible to new files with new formats being added. If new files were added to one 
of the source file directories with a slightly different naming format, it would require going deep into 
several scripts in order to change how they extracted the subject name each new file was associated with, it's 
D/HC label, etc. This process ends up just adding more 'if...else' clauses to many already-cluttered parts of 
the scripts.

2.) As a result of having to change numerous things in several scripts, the process was more prone to human 
error. For example, as a result of a small oversight and not correctly reading the 'D' part of a file name that 
corresponded to subjects with 'D' in their subject name (e.g. 'D5'), the script was incorrectly interpreting the 
D/HC label as being 'HC' rather than 'D' like it should have been; hence, the model was trained incorrectly due 
to labelling sequences incorrectly. In comparison, if we would have used 'file_renamer.py' from the beginning, 
we would have easily spotted any files that have been renamed incorrectly and correct them before other scripts 
had the chance to misinterpret their labels.



------ How it works ------


The basic operation of the 'file_renamer.py' script can be summarized as follows:

1.) Read in the name of a source directory of .mat files of which we wish to standardize the names.

2.) Gets the names of all .mat files within the directory and divides them into one of two categories: 
'files_kept' (i.e. the vast majority of files which we don't want to remove) and 'files_to_delete' (files which 
we want to remove from the directory. Note that this is only for certain files that have been previously 
determined to be too large, too small, or not 'relevant' files to either training or testing models (for example, 
files that contain 'AllTasks' in the 'allmatfiles' source directory, as these contain the same information as 
the other files in the directory but concatenated together for a single subject, so there's no need to use 
these as well as the others.

3.) Based on the source directory name, apply a set of regular expression ('regex') rules to each file name 
that are in 'files_kept'. These are unique to each directory, as there are some things that we need to check for 
in some directories but not in other. These regular expressions are a set of substitutions: they search the 
file name for a certain quality and, if it finds it, replaces it with another, before using this new string as 
the basis for the next regex. These regexes include: replacing non-capitalized subject names to capitalized 
versions (e.g. changing 'd4-003.mat' to 'D4-003.mat'), replacing 'NSA' with 'NSAA when found in a file name, 
changing instances of '-6MW.mat' to '-6MinWalk.mat' (as the type of activities they contain is the same whether 
it was sourced from '6minwalk-matfiles' or 6MW-matFiles'), and so on

4.) With this new list of file names that we are to change 'files_kept' to, we first remove the files within 
the source directory based on the file names within 'files_to_delete' and then, for each name in 'files_kept' 
and its corresponding name in 'new_files_names' replace the name of the file in the former with the name 
in the latter. The result is that all of the files within the specified source directory are automatically 
changed based on the standard we predefined.




------ Additional notes ------

However, it's important to note that this script is not intended to be run more than once, and only at the 
beginning. Hence, it should be executed before any of the other scripts like 'comp_stat_vals.py' or 
'ext_raw_measures.py' are used. This is because these scripts use the names of the files they are sourced from 
to create new files with names based on their source names; hence, for 'file_renamer.py' to be useful, they 
should be used prior to other files being created that are based on the files that 'file_renamer.py' wishes 
to rename.

To this end, 'file_renamer.py' is only needed to be used once. For this reason, it's also included within 
'setup.cmd' as part of the setup process and is applied before any of the other scripts for the above reason.