REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=2
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=3
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=4
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=5
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=6
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=7
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=8
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=9
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=10
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=11
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=12
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=13
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=14
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=15
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=16
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=17








REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngleXZY all dhc --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngleXZY all overall --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA jointAngleXZY all acts --seq_len=600 --seq_overlap=0.98 --discard_prop=0.9 --epochs=20 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.98 --epochs=100 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.98 --epochs=100 --leave_out=D3 --balance=down
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.98 --epochs=100 --leave_out=D3 --balance=down

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





REM python ..\rnn.py allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20 --leave_out=D11




REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY,AD D3 --use_balanced=down
REM python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY,AD D3 --use_balanced=up
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













REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3
REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9
REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11
REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17
REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC2
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC2
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC2
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC2
python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6







