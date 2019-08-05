
REM Note that 'model_predictions_set_3.cmd' must have been executed first in order to execute the below lines correctly
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D3 --batch
python ..\model_predictor.py NSAA position D3 --batch
python ..\model_predictor.py NSAA sensorMagneticField D3 --batch
python ..\model_predictor.py NSAA jointAngle D3 --batch
python ..\model_predictor.py NSAA AD D3 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D9 --batch
python ..\model_predictor.py NSAA position D9 --batch
python ..\model_predictor.py NSAA sensorMagneticField D9 --batch
python ..\model_predictor.py NSAA jointAngle D9 --batch
python ..\model_predictor.py NSAA AD D9 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D11 --batch
python ..\model_predictor.py NSAA position D11 --batch
python ..\model_predictor.py NSAA sensorMagneticField D11 --batch
python ..\model_predictor.py NSAA jointAngle D11 --batch
python ..\model_predictor.py NSAA AD D11 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD D17 --batch
python ..\model_predictor.py NSAA position D17 --batch
python ..\model_predictor.py NSAA sensorMagneticField D17 --batch
python ..\model_predictor.py NSAA jointAngle D17 --batch
python ..\model_predictor.py NSAA AD D17 --batch
python ..\model_predictor.py NSAA position,sensorMagneticField,jointAngle,AD HC6 --batch
python ..\model_predictor.py NSAA position HC6 --batch
python ..\model_predictor.py NSAA sensorMagneticField HC6 --batch
python ..\model_predictor.py NSAA jointAngle HC6 --batch
python ..\model_predictor.py NSAA AD HC6 --batch