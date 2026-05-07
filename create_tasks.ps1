$PYTHON = "C:\Yaariok\.venv\Scripts\python.exe"

# Create Yaariok_KB_On task
schtasks /create /tn "Yaariok_KB_On" `
  /tr "`"$PYTHON`" C:\Yaariok\kb_control.py on" `
  /sc once /st 00:00 /ru SYSTEM /rl HIGHEST /f

# Create Yaariok_KB_Off task
schtasks /create /tn "Yaariok_KB_Off" `
  /tr "`"$PYTHON`" C:\Yaariok\kb_control.py off" `
  /sc once /st 00:00 /ru SYSTEM /rl HIGHEST /f

"Tasks created OK" | Out-File "C:\Yaariok\task_setup_result.txt"
