python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --pca=60 --batch
python ..\rnn.py NSAA position,sensorMagneticField,jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --pca=60 --batch

python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --use_ft_concat --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --use_ft_concat --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --use_ft_concat --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --use_ft_concat --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --use_ft_concat --batch