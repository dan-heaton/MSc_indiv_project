REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --batch

REM Note that 'model_predictions_set_3.cmd' must have been executed first in order to execute the below lines correctly
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D3 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D3 --batch --use_seen
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D9 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D9 --batch --use_seen
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D11 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D11 --batch --use_seen
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D17 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D17 --batch --use_seen
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD HC6 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD HC6 --batch --use_seen