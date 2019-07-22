---------------------		data_balancer.py Overview and explanation		---------------------


------ Motivation ------

One of the inherent problems with the dataset is the lack of 'variance' within the subjects for their overall 
scores. This is mainly a feature of how the NSAA scores are conducted and the variation of severity of Duchenne 
muscular dystrophy within the patients. As the individual activity scores range from 0 (can't complete the activity 
at all) to 2 (completes it perfectly) and as there are 17 activites in total, the overall cumulative score ranges 
from 0 to 34. However, in reality, most patients in the study have scores ranging between 15 - 24 for moderate 
Duchenne. When it comes to training a network on the subjects' data and testing it on new files, this causes a 
problem if the subject has a particularly low overall NSAA score (e.g. 3). In other words, the slight lack of 
variation in the data we have available may slightly limit the potential on new subjects with particularly extreme 
cases of Duchenne muscular dystrophy. Thus, this is an important aspect to cover when we wish to improve 
generalization performance of the models to new subjects.

A classic way in machine learning of helping to get around this is in data balancing. This is traditionally done 
for classification problems rather than regression problems, as we are doing here. However, we get around this by, 
for the purposes of rebalancing the data set, considering overall NSAA scores as class labels rather than scores to 
be regressed on. There are two ways we consider here to balance our data, which are outlined below:

Consider a dataset of 10 sequences of data (i.e. 2D structures of data of shape 'sequence length' x '# features') 
with scores: [3, 15, 15, 15, 20, 20, 34, 34, 34, 34]. We have 2 ways of approaching this:

	Downsampling: counts the frequency of each number in the list and finds the lowest frequency; in the above 
	case, it is '1' (as there is only 1 '3' in the list). Next, for each of the labels in the list above, we 
	randomly select '1' sample of each label in the list and, more importantly, the label's corresponding 
	'x' value (i.e. a single sequence). Thus, we are reduced to a list of 4 sequences and with a label list 
	of [3, 15, 20, 34] (note that there is only 1 of each sample because there was originally 1 '3' label.
	Hence, we now have a much smaller list, but an even spread of 'y' values for the samples we have remaining.

	Upsampling: we start off the same, with finding the frequency of each number in the list, but this time 
	considering the highest frequency in the list. In the above case, this would be '4', as there are 4 34's 
	in the list. Next, for each label value in the list, we randomly sample a 'y' and corresponding 'x' value 
	(being a sequence) a total of '4' times for each label. For example, for the '15' labels (i.e. 3 sequences 
	and 3 '15' labels), we randomly pick a pair of 'x' and 'y' values from the 3 available and do this '4' 
	times. Thus, we end up with a much larger list of [3, 3, 3, 3, 15, 15, 15, 15, 20, 20, 20, 20, 34, 34, 34, 34] 
	of 'y' values with corresponding 'x' values (sequences).

Upsampling has the advantage of not discarding any of the data that has been given to us; however, it means that 
many samples are repeatedly used as 'new' samples, which may lead to unpredicable training results, along with an 
inflated data set may being more challenging to train on. Downsampling, meanwhile, might give better generalization 
results than non-resampled data while being a smaller data set (thus making it quicker to train models that 
achieve better results), but the discarding of data points might leave important insights from the data out of the 
training process.




------ How it works ------

The script contains 3 functions: 'ext_label_dist', 'downsample', and 'upsample'. The last two functions are 
more-or-less identical to their respective algorithms that are outlined above, with a few implementation details 
differing but the overall ideas being the sample; hence, we won't repeat the more-or-less same algorithm here. 
Instead, it's worth considering how each of the functions are used. The script is never run directly, but rather 
serves simply as a storage place for several functions that are fetched by 'rnn.py'; hence, it's instead useful 
to consider exclusively how 'rnn.py' use the functions. Also note that these are only run by the 'rnn.py' script 
if the '--balance' optional argument is provided.

It's also worth noting the distinction between 'y_data' and 'y_data_balance' when used as parameters for 
'downsample' and 'upsample'. 'y_data' might be, depending on the output type that we are training towards (e.g. 
D/HC classification, overall NSAA score, or single act scores) a list of 1's and 0's, a list of values between 0 
and 34, or a list of lists of 17 values between 0 and 2. Hence, we want a unified way of rebalancing the data that 
is irrespective of the form that 'y_data' takes. Hence, 'y_data_balance' will *always* be the overall NSAA scores 
for the corresponding 'x_data'; if, for 'rnn.py', the 'choice' arg is 'overall', then this will be exactly the 
same as 'y_data', but for others it will contain the overall NSAA scores that are corresponding to the 'x' and 
'y' values. The 'y_data_balance' is then used in the algorithms outlined above to find the indeces of 'x_data' and 
'y_data' to select to create the new lists of data.


The functions of 'data_balancer.py' and how they are used by 'rnn.py' are as follows:

-- ext_label_dist(): for each file that the 'rnn.py' model is training on, read in the 'nsaa_6mw_info.xlsx' file, 
finds the relevant row in the table corresponding to the file name in question, and returns the overall NSAA score 
for this file name. This is then used as the label for each of the sequences that are extracted from the file in 
question, and the process is then repeated for every other file in the source directory, 'dir'.

-- downsample(): if the '--balance' argument is set as 'up', then this function is called that takes in the 
'x_data' and 'y_data' created from sequences (as 'rnn.py' would normally create) and the additional 'y_data_balance' 
that we have create additionally to use to balance the script, and from these downsamples the data and produces 
two new lists of 'new_x_data' and 'new_y_data' via the algorithm outlined above.

-- upsample(): called in the same way as 'downsample()' but via '--balance=up', while taking in the same arguments 
but instead using the algorithm for upsampling as described above.

With the 'ext_label_dist()' and either 'downsample()' or 'upsample()' having been run the requisite number of 
times ('ext_label_dist()' once for every file in the source directory, 'dir', and only once for either of the 
other two), this data then replaces the original 'x_data' and 'y_data' in 'rnn.py', prints the new balanced shapes 
to the user, adds several output strings to be printed at the end of the script's running to show the before and 
after data balancing for the distribution of labels, and the execution of 'rnn.py' subsequently continues as usual.


