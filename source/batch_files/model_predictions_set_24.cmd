python ..\rnn.py allmatfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20
python ..\rnn.py allmatfiles jointAngle all overall --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20
python ..\rnn.py allmatfiles jointAngle all acts --seq_len=600 --seq_overlap=0.1 --discard_prop=0.9 --epochs=20

python ..\test_altdirs.py NSAA allmatfiles jointAngle
python ..\graph_creator.py model_preds_altdirs NSAA allmatfiles