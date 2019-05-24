@echo off
if %ERRORLEVEL% == 1 exit /b
if %ERRORLEVEL% == 2 exit /b

setlocal
set directory=%1
set file_type=%2
set file_name=%3
set split_size=%4
set ft_red_choice=%5
set extract_csv=%6

@echo ---------- Running 'matfiles_analysis.py...' ----------
python matfiles_analysis.py %directory% %file_type% %file_name% --split_size% %=% %%split_size% %extract_csv%

@echo ---------- Running 'ft_red_nsaa.py...' ----------
REM only run the feature selection/reduction script if it's not a direct-from-JA file (i.e. with 'matfiles_analysis' using --extract_csv).
if [%6]==[] (python ft_red_nsaa.py %directory% %file_type% %file_name% %ft_red_choice%)

@echo ---------- Running 'basic_rnn.py...' ----------
if not [%6]==[] (python basic_rnn.py %file_type% %file_name% --split_size% %=% %%split_size% --nss)


endlocal