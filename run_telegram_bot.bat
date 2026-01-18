@echo off
REM Move to the directory where this .bat file lives (project root)
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate
) else (
    echo WARNING: .venv not found. Using system Python.
)

REM Run the Telegram command client
python src\ops\telegram_command_client.py
