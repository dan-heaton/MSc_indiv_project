REM python ..\rnn.py NSAA position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=20 --no_testset --batch
REM python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=20 --no_testset --batch

REM python ..\rnn.py NMB position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch
REM python ..\rnn.py NMB AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --batch

REM python ..\test_altdirs.py NMB NSAA sensorMagneticField --batch
REM python ..\test_altdirs.py NMB NSAA position,sensorMagneticField,jointAngle,AD --batch
REM python ..\test_altdirs.py NSAA NMB sensorMagneticField --batch
REM python ..\test_altdirs.py NSAA NMB position,sensorMagneticField,jointAngle,AD --batch
python ..\graph_creator.py model_preds_altdirs NMB NSAA sensorMagneticField --save_img --no_display --batch
python ..\graph_creator.py model_preds_altdirs NMB NSAA position,sensorMagneticField,jointAngle,AD --save_img --no_display --batch
python ..\graph_creator.py model_preds_altdirs NSAA NMB sensorMagneticField --save_img --no_display --batch
python ..\graph_creator.py model_preds_altdirs NSAA NMB position,sensorMagneticField,jointAngle,AD --save_img --no_display --batch