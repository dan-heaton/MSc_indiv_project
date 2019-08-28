python ..\rnn.py NSAA,NMB position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch
python ..\rnn.py NSAA,NMB AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=3 --no_testset --batch