@echo off

setlocal
set directory=NSAA
set file_path=%1



@echo.
@echo ---------- Copying '%file_path%' to '%directory%'... ----------
@echo.
python ..\file_mover.py %directory% %file_path%

REM Sets the 'file_name' variable to the file name of the complete path (i.e. excluding directories)
For %%A in ("%file_path%") do (set file_name=%%~nxA)

@echo.
@echo ---------- Running 'ext_raw_measures.py...' ----------
@echo.
python ..\ext_raw_measures.py %directory% %file_name% jointAngle,sensorMagneticField,position

@echo.
@echo ---------- Running 'comp_stat_vals.py...' ----------
@echo.
if %directory% == NSAA python ..\comp_stat_vals.py %directory% AD %file_name% --split_size=1


@echo.
@echo ---------- Running 'ft_sel_red.py...' ----------
@echo.
if %directory% == NSAA python ..\ft_sel_red.py %directory% AD %file_name% pca --num_features% %=%30 --no_normalize


@echo.
@echo ---------- Running 'model_predictor.py...' ----------
@echo.
REM Runs the 'model_predictor.py' on the file using only the sensor magnetic field measurement then using all measurements
python ..\model_predictor.py %directory% sensorMagneticField %file_name% --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py %directory% position,sensorMagneticField,jointAngle,AD %file_name% --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch