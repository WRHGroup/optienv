@echo off
REM Path to Julia executable
set JULIA=C:\Users\Mohammed Basheer\.julia\juliaup\julia-1.11.6+0.x64.w64.mingw32\bin\julia.exe

REM Always change to the folder where this .bat file is located
cd /d %~dp0

REM Run Julia with the run_wflow.jl script and the TOML config (one level up)
"%JULIA%" ".\run_wflow.jl" ".\wflow_sbm.toml"
