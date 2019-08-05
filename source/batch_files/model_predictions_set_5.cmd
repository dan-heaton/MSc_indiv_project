python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D3 --use_frc --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9 --use_frc --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9 --use_frc --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D9 --use_frc --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D11 --use_frc --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17 --use_frc --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17 --use_frc --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=D17 --use_frc --batch
python ..\rnn.py NSAA AD all dhc --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6 --use_frc --batch
python ..\rnn.py NSAA AD all overall --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6 --use_frc --batch
python ..\rnn.py NSAA AD all acts --seq_len=10 --seq_overlap=0.9 --epochs=100 --leave_out=HC6 --use_frc --batch



REM Note that 'model_predictions_set_3.cmd' must have been executed first in order to execute the below 5 lines correctly
python ..\model_predictor.py NSAA AD D3 --batch
python ..\model_predictor.py NSAA AD D9 --batch
python ..\model_predictor.py NSAA AD D11 --batch
python ..\model_predictor.py NSAA AD D17 --batch
python ..\model_predictor.py NSAA AD HC6 --batch



python ..\model_predictor.py NSAA AD D3 --use_frc --batch
python ..\model_predictor.py NSAA AD D9 --use_frc --batch
python ..\model_predictor.py NSAA AD D11 --use_frc --batch
python ..\model_predictor.py NSAA AD D17 --use_frc --batch
python ..\model_predictor.py NSAA AD HC6 --use_frc --batch
