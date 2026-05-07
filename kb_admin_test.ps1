$ns = "root\WMI"
$cls = [wmiclass]"\\.\$ns:hpCpsPubGetSetCommand"
$inP = $cls.GetMethodParameters("hpCpsPubSetCommand")
$inP["Command"] = "keyboard"
$inP["DataSizeIn"] = [uint32]1
$inP["hpqBDataIn"] = [byte[]]@([byte]3)
$inP["SignIn"] = [uint32]0
try {
    $r = $cls.InvokeMethod("hpCpsPubSetCommand", $inP, $null)
    "ReturnCode: $($r.ReturnCode)" | Out-File "C:\Yaariok\kb_test_result.txt"
} catch {
    "Error: $_" | Out-File "C:\Yaariok\kb_test_result.txt"
}
