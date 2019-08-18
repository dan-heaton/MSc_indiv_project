python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D3 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D9 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D11 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=D17 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch
python ..\rnn.py NSAA,allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --leave_out=HC6 --batch

REM python ..\model_predictor.py NSAA jointAngle,AD D3 --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D3 --add_dir=allmatfiles --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D9 --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D9 --add_dir=allmatfiles --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D11 --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D11 --add_dir=allmatfiles --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D17 --batch
REM python ..\model_predictor.py NSAA jointAngle,AD D17 --add_dir=allmatfiles --batch
REM python ..\model_predictor.py NSAA jointAngle,AD HC6 --batch
REM python ..\model_predictor.py NSAA jointAngle,AD HC6 --add_dir=allmatfiles --batch