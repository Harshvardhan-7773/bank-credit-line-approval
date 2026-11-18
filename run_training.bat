@echo off
title ApexBank Credit Line Model Retraining
echo ====================================================================
echo Retraining Dual-Stage Underwriting Models...
echo ====================================================================
"C:\Users\hsing\AppData\Local\Programs\Python\Python314\python.exe" src\data_generator.py
"C:\Users\hsing\AppData\Local\Programs\Python\Python314\python.exe" src\train_models.py
"C:\Users\hsing\AppData\Local\Programs\Python\Python314\python.exe" tests\test_pipeline.py
echo ====================================================================
echo Retraining Complete!
echo ====================================================================
pause
