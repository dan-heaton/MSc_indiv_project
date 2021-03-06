python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=HC6 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=HC6 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_allmatfiles=3 --leave_out=HC6 --batch

python ..\model_predictor.py NSAA jointAngle D3 --batch
python ..\model_predictor.py NSAA jointAngle D3 --add_dir=allmatfiles --batch
python ..\model_predictor.py NSAA jointAngle D9 --batch
python ..\model_predictor.py NSAA jointAngle D9 --add_dir=allmatfiles --batch
python ..\model_predictor.py NSAA jointAngle D11 --batch
python ..\model_predictor.py NSAA jointAngle D11 --add_dir=allmatfiles --batch
python ..\model_predictor.py NSAA jointAngle D17 --batch
python ..\model_predictor.py NSAA jointAngle D17 --add_dir=allmatfiles --batch
python ..\model_predictor.py NSAA jointAngle HC6 --batch
python ..\model_predictor.py NSAA jointAngle HC6 --add_dir=allmatfiles --batch