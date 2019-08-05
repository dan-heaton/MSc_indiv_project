python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngleXZY all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngleXZY all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngleXZY all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --batch

python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D3 --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D3 --batch
python ..\model_predictor.py NSAA position D3 --batch
python ..\model_predictor.py NSAA sensorMagneticField D3 --batch
python ..\model_predictor.py NSAA jointAngle D3 --batch
python ..\model_predictor.py NSAA jointAngleXZY D3 --batch
python ..\model_predictor.py NSAA AD D3 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D11 --use_seen --batch
python ..\model_predictor.py NSAA position D11 --batch
python ..\model_predictor.py NSAA sensorMagneticField D11 --batch
python ..\model_predictor.py NSAA jointAngle D11 --batch
python ..\model_predictor.py NSAA jointAngleXZY D11 --batch
python ..\model_predictor.py NSAA AD D11 --batch