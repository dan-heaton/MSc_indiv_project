@echo off

setlocal
set directory=%1
set file_name=%2
set altdirs=%3

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


@echo.
@echo ---------- Running 'model_predictor.py...' ----------
@echo.
python ..\model_predictor.py %directory% %measures% %file_name% --alt_dirs=%altdirs%