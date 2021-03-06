python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D9 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D9 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D9 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D17 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D17 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=D17 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=HC6 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=HC6 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --leave_out=HC6 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6 --batch
python ..\rnn.py NSAA position all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --batch
python ..\rnn.py NSAA sensorMagneticField all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --batch
python ..\rnn.py NSAA jointAngle all indiv --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=100 --batch
python ..\rnn.py NSAA AD all indiv --seq_len=10 --seq_overlap=0.9 --epochs=100 --batch



python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=1 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=2 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=3 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=4 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=5 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=6 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=7 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=8 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=9 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=10 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=11 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=12 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=13 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=14 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=15 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=16 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=17 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=1 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=2 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=3 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=4 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=5 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=6 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=7 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=8 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=9 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=10 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=11 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=12 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=13 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=14 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=15 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=16 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=17 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=1 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=2 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=3 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=4 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=5 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=6 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=7 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=8 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=9 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=10 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=11 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=12 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=13 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=14 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=15 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=16 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=17 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=1 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=2 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=3 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=4 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=5 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=6 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=7 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=8 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=9 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=10 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=11 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=12 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=13 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=14 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=15 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=16 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=17 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=1 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=2 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=3 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=4 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=5 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=6 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=7 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=8 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=9 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=10 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=11 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=12 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=13 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=14 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=15 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=16 --use_indiv --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=17 --use_indiv --batch

python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=1 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=2 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=3 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=4 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=5 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=6 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=7 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=8 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=9 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=10 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=11 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=12 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=13 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=14 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=15 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=16 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D3 --single_act=17 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=1 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=2 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=3 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=4 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=5 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=6 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=7 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=8 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=9 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=10 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=11 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=12 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=13 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=14 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=15 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=16 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D9 --single_act=17 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=1 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=2 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=3 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=4 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=5 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=6 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=7 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=8 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=9 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=10 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=11 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=12 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=13 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=14 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=15 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=16 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D11 --single_act=17 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=1 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=2 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=3 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=4 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=5 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=6 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=7 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=8 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=9 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=10 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=11 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=12 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=13 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=14 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=15 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=16 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle D17 --single_act=17 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=1 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=2 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=3 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=4 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=5 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=6 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=7 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=8 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=9 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=10 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=11 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=12 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=13 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=14 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=15 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=16 --use_indiv --use_seen --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle HC6 --single_act=17 --use_indiv --use_seen --batch

REM Note that the below two sets of integer arguments to 'graph_creator.py' may need to be changed to the correct row numbers in 'model_predictions.csv'
python ..\graph_creator.py model_preds_trues_preds 641 707 --batch --save_img --no_display
python ..\graph_creator.py model_preds_trues_preds 708 774 --batch --save_img --no_display