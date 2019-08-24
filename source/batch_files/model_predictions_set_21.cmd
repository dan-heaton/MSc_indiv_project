python ..\rnn.py NSAA,NMB position all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB position all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB position all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB sensorMagneticField all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB jointAngle all dhc --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB jointAngle all overall --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB jointAngle all acts --seq_len=600 --seq_overlap=0.9 --discard_prop=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch
python ..\rnn.py NSAA,NMB AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=20 --balance_nmb=2 --no_testset --leave_out_version=V2 --batch

python ..\model_predictor.py NSAA sensorMagneticField D4V2-005 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D4V2-005 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D4V2-010 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D4V2-010 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D5V2-010 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D5V2-010 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D5V2-011 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D5V2-011 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D6V2-007 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D6V2-007 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D6V2-008 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D6V2-008 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D7V2-002 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D7V2-002 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D7V2-004 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D7V2-004 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA sensorMagneticField D7V2-03 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D7V2-03 --add_dir=NMB --combine_preds --no_testset --leave_out_version=V2 --new_subject --batch