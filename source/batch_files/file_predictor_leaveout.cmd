@echo off

setlocal
set directory=%1
set file_name=%2

if %ERRORLEVEL% == 1 exit /b
if %ERRORLEVEL% == 2 exit /b


if %directory% == NSAA set file_type=jointAngle,sensorMagneticField,position
if %directory% == NSAA set measures=AD,jointAngle,sensorMagneticField,position
if %directory% == allmatfiles set file_type=jointAngle
if %directory% == allmatfiles set measures=jointAngle


@echo.
@echo ---------- Running 'ext_raw_measures.py...' ----------
@echo.
python ..\ext_raw_measures.py %directory% %file_name% %file_type%


@echo.
@echo ---------- Running 'comp_stat_vals.py...' ----------
@echo.
if %directory% == NSAA python ..\comp_stat_vals.py %directory% AD %file_name% --split_size=1


@echo.
@echo ---------- Running 'ft_sel_red.py...' ----------
@echo.
if %directory% == NSAA python ..\ft_sel_red.py %directory% AD %file_name% pca --num_features% %=%30 --no_normalize


python ..\rnn.py %directory% position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python ..\rnn.py %directory% AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%
python ..\rnn.py %directory% AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%
python ..\rnn.py %directory% AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%


@echo.
@echo ---------- Running 'model_predictor.py...' ----------
@echo.
python ..\model_predictor.py %directory% %measures% %file_name%