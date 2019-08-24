
---------------------		file_mover.py Overview and explanation		---------------------


------ Motivation ------

To enable the working of certain batch scripts, it became a necessity to build into the batch files the 
ability to relocate files that are located anywhere on a user's PC to the proper sub directory of the 
local directory in order to have the data pipeline run properly. For example, if there was a source '.mat' 
file for 'D9V2' subject as an NSAA file (i.e. the second NSAA assessment done for subject 'D9') located 
somewhere on a user's PC, we wish to be able to copy it over to the <local directory>\NSAA\matfiles\ 
subdirectory. However, while we are able to do this potentially in a batch file via the 'move' command, we 
also wish to be able to change the location of where to copy the file to depend on the type of file 
we are working with (e.g. if the file is an NMB file, it would be placed in a different location within the 
local directory than if it was an NSAA file); additionally, we also wish to make use of the 'local_dir' 
variable stored in 'settings.py' so one wouldn't have to modify a variable within a batch file if the 
local directory location was changed.

For the above reasons, it was evident that it was simply easier to implement the 'move file' functionality 
to its own separate Python script. The intention, however, is to only ever use this file as part of a 
batch file (e.g. 'assess_nsaaV2_file.cmd') as the first step in placing a file in the correct location 
to be used within the data pipeline.



------ How it works ------

As this is a short script with a singular purpose, it's worth outlining the simple steps as the program 
runs in a procedural manner.

1.) Takes in as arguments the name of the directory within the local directory to place the file within 
based on the type of file (e.g. 'NMB', 'NSAA', '6minwalk-matfiles', etc.) and the complete or local path 
(relative to the <project directory>\source\batch_files\ directory) to the file we wish to move.

2.) Checks for argument validity for the 'dir' argument and, given it is one of the allowed options, 
add the strings to the 'local_dir' variable based on the 'dir' argument so that 'local_dir' now points to 
the correct 'inner' directory to store the copied source '.mat' file in.

3.) Attempts to copy the file given as the argument to the program to the new value of 'local_dir', and 
throws an exception if it cannot locate the file by the path given.