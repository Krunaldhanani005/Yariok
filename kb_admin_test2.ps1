$ns = "root\WMI"
$cls = [wmiclass]"\\.\$ns:hpCpsPubGetSetCommand"
$results = @()

# Try the hpCpsPubGetCommand first to list valid commands
$inPGet = $cls.GetMethodParameters("hpCpsPubGetCommand")
$cmds = @("keyboard", "0x20003", "131075", "kbl", "backlight", "omenKeyboard", "platform")
foreach ($cmd in $cmds) {
    try {
        $inPGet["Command"] = $cmd
        $inPGet["SignIn"]  = ""
        $r = $cls.InvokeMethod("hpCpsPubGetCommand", $inPGet, $null)
        $results += "'$cmd' GET OK: $($r.ReturnValue)"
    } catch {
        $results += "'$cmd' GET Error: $($_.Exception.Message.Substring(0,[Math]::Min(60,$_.Exception.Message.Length)))"
    }
}
$results | Out-File "C:\Yaariok\kb_test_result.txt"
