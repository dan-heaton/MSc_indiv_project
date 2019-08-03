@echo off

if %ERRORLEVEL% == 1 exit /b
if %ERRORLEVEL% == 2 exit /b


REM install all the necessary packages to run all of the scripts
pip install numpy==1.16.3
pip install tensorflow-gpu==1.13.1
pip install scikit-learn==0.20.0
pip install protobuf==3.6.0
pip install scipy==1.1.0
pip install pandas==0.24.2
pip install matplotlib==3.0.0
pip install pyexcel=0.5.13


REM run the necessary scripts to setup the files necessary for the RNN and the model predictor
python ..\file_renamer.py NSAA
python ..\file_renamer.py 6minwalk-matfiles
python ..\file_renamer.py 6MW-matFiles
python ..\file_renamer.py allmatfiles
python ..\mat_act_div.py all
python ..\mat_act_div.py all --concat_act_files
python ..\comp_stat_vals.py NSAA AD all --split_size=1
python ..\comp_stat_vals.py NSAA AD all --split_size=1 --single_act
python ..\comp_stat_vals.py NSAA AD all --split_size=1 --single_act_concat
python ..\comp_stat_vals.py 6minwalk-matfiles AD all --split_size=1
python ..\comp_stat_vals.py 6minwalk-matfiles DC all --extract_csv
python ..\comp_stat_vals.py 6minwalk-matfiles JA all --extract_csv
python ..\comp_stat_vals.py 6MW-matFiles AD all --split_size=1
python ..\ext_raw_measures.py NSAA all all
python ..\ext_raw_measures.py NSAA all all --single_act
python ..\ext_raw_measures.py NSAA all all --single_act_concat
python ..\ext_raw_measures.py 6minwalk-matfiles all all
python ..\ext_raw_measures.py 6MW-matFiles all all
python ..\ext_raw_measures.py allmatfiles all jointAngle
python ..\ft_sel_red.py NSAA AD all pca --num_features=30 --no_normalize
python ..\ft_sel_red.py 6minwalk-matfiles AD all pca --num_features=30 --no_normalize 
python ..\ft_sel_red.py 6MW-matFiles AD all pca --num_features=30 --no_normalize


@echo ---------- Setup successfully completed ----------
