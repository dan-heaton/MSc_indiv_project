python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6 --batch



python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=1 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=2 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=3 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=4 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=5 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=6 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=7 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=8 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=9 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=10 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=11 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=12 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=13 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=14 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=15 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=16 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D3 --single_act=17 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=1 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=2 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=3 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=4 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=5 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=6 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=7 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=8 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=9 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=10 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=11 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=12 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=13 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=14 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=15 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=16 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D9 --single_act=17 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=1 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=2 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=3 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=4 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=5 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=6 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=7 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=8 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=9 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=10 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=11 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=12 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=13 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=14 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=15 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=16 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D11 --single_act=17 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=1 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=2 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=3 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=4 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=5 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=6 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=7 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=8 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=9 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=10 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=11 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=12 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=13 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=14 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=15 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=16 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY D17 --single_act=17 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=1 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=2 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=3 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=4 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=5 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=6 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=7 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=8 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=9 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=10 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=11 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=12 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=13 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=14 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=15 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=16 --use_indiv
python model_predictor.py NSAA position,sensorMagneticField,jointAngle,jointAngleXZY HC6 --single_act=17 --use_indiv