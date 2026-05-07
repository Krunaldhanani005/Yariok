$PYTHON = "C:\Yaariok\.venv\Scripts\python.exe"
& $PYTHON "C:\Yaariok\kb_control.py" on 2>&1 | Out-File "C:\Yaariok\kb_test_output.txt"
