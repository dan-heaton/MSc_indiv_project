






REM python model_predictor.py allmatfiles jointAngle D11-004



REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngleXZY all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngleXZY all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA jointAngleXZY all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --use_frc



REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc

REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc


REM python model_predictor.py NSAA AD D3 --use_frc
REM python model_predictor.py NSAA AD D11 --use_frc



REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=D3
REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=D9 --balance=down
REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=D11 --balance=down
REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=D17 --balance=down
REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=HC2 --balance=down
REM python ..\rnn.py cnn_data sensorFreeAcceleration,sensorOrientation all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=10 --leave_out=HC6 --balance=down


REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngleXZY all dhc --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngleXZY all overall --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA jointAngleXZY all acts --seq_len=600 --seq_overlap=0.5 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.5 --epochs=100 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.5 --epochs=100 --leave_out=D3 --balance=up
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.5 --epochs=100 --leave_out=D3 --balance=up






