$r = schtasks /run /tn "Yaariok_KB_On" 2>&1
"Run result: $r" | Out-File "C:\Yaariok\kb_run_test.txt"
Start-Sleep -Seconds 4
"Task triggered, check keyboard" | Out-File "C:\Yaariok\kb_run_test.txt" -Append
