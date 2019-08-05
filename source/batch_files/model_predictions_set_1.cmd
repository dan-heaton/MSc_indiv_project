python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py 6minwalk-matfiles jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py 6minwalk-matfiles jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch
python ..\rnn.py 6minwalk-matfiles jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --batch

python ..\test_altdirs.py allmatfiles NSAA,6minwalk-matfiles jointAngle
python ..\graph_creator.py model_preds_altdirs allmatfiles NSAA,6minwalk-matfiles