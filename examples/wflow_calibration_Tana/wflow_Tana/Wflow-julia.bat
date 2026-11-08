@echo off
REM Path to Julia executable
set JULIA=C:\Users\Addis\.julia\juliaup\julia-1.11.6+0.x64.w64.mingw32\bin\julia.exe

REM Run from the folder containing this .bat file
cd /d %~dp0

REM Stop if the TOML config is missing
if not exist "wflow_sbm.toml" (
    echo wflow_sbm.toml not found in %cd%
    pause
    exit /b 1
)

REM Run Wflow with the TOML config in this folder
"%JULIA%" -e "using Wflow; println(\"Running Wflow with config: \", ARGS[1]); Wflow.run(ARGS[1])" "wflow_sbm.toml"