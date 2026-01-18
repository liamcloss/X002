@echo off
setlocal enabledelayedexpansion

REM Move to the directory where this .bat file lives (project root)
cd /d "%~dp0"

REM Simple log directory
if not exist logs mkdir logs

:loop
echo ============================================== >> logs\bot.log
echo Bot starting at %DATE% %TIME% >> logs\bot.log

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate
) else (
    echo WARNING: .venv not found. Using system Python. >> logs\bot.log
)

REM Run the Telegram command client
python src\ops\telegram_command_client.py >> logs\bot.log 2>&1

REM If we get here, Python exited (crash or clean exit)
echo Bot exited at %DATE% %TIME% >> logs\bot.log
echo Restarting in 15 seconds... >> logs\bot.log

timeout /t 15 > nul
goto loop
