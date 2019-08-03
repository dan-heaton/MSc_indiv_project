

---------------------		settings.py: Overview and explanation		---------------------

The purpose of this file is to hold many of the the variables that are used throughout the rest of the script. 
In particular, there are many variable names (such as 'source_dir') that hold the same values throughout all of 
the scripts. These variables contain values that include directory sources paths, paths to certain files that 
scripts output information to, lists of sensor names that have been given to us via the 'MVN User Manual', and 
so on; the common factor is that they are all referenced as being the same values across several different scripts 
and are thus interpreted as system constants.


In storing these values in a separate file, we achieve three things:

1.) It reduces the amount of overall 'clutter' within the scripts, especially when we need to reference large 
variables such as those holding large lists of strings, which makes the scripts themselves both easier to debug 
and to maintain.

2.) For variables that are supposed to remain static, it reduces the possibility of accidentally changing them to 
suit the script they are currently being referenced in. For example, we are less likely to accidentally change the 
name of one of the 'raw_measurements' when they are only accessed in other scripts and not modified and, if one is 
changed in 'settings', then this change is reflected out to all other scripts in the same way (e.g. preventing 
two scripts from each having their own versions of 'raw_measurements', which could cause conflict in manipulating 
output files.

3.) If they are required to change for whatever reason (e.g. if a new user has their 'local_dir' in a different 
location to the default value, or if the batch size to be used across numerous scripts is modified to be something 
else), then it's much easier to do so in a single 'settings' script rather than tracking down and modifiying each 
respective variable in each script.




To access these values, each of the scripts calls the necessary variables from settings in the 'import' section
of the script. The idea of scripts only importing the variables that it needs was that it enhances clarity (i.e. 
if 'from settings import *' was used, we wouldn't as easily be able to see that 'local_dir' comes from 'settings' 
as if we used 'from settings import local_dir'). Additionally, it's also recommended that any user using this 
project and setup for the first time should first examine the relevant path names (such as 'local_dir', 
'results_path', etc.) to ensure that the source mat files are contained in the expected location, the scripts can 
access the necessary output .xlsx and .csv files, and so on.

