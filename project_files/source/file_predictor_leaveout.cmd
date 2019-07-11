@echo off

setlocal
set directory=%1
set file_name=%2


python rnn.py %directory% position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngleXZY all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngleXZY all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% jointAngleXZY all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=%file_name%
python rnn.py %directory% AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%
python rnn.py %directory% AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%
python rnn.py %directory% AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=%file_name%


python model_predictor.py %directory% position,sensorMagneticField,jointAngle,jointAngleXZY,AD %file_name%