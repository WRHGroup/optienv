@echo off
call C:\Users\Addis\anaconda3\Scripts\activate.bat hydromt-wflow-py310
cd /d %~dp0..
set DASK_SCHEDULER=synchronous
set WRAPPER_WARMUP_YEARS=0
set RESUME=
if exist "results\checkpoint.npz" set RESUME=--resume-latest

optienv search ^
  -c C:\...\wflow_calibration_East_Pine\run_East_Pine.json ^
  --algo nsga2 ^
  -j 4 --seed 7 ^
  --label-columns ^
  --no-save-final-csvs ^
  --work-root C:\...\wflow_calibration_East_Pine ^
  --checkpoint-every 1 %RESUME%
pause
