$result = ""
# Try hpCpsPubSetCommand via Invoke-WmiMethod
try {
    $r = Invoke-WmiMethod -Namespace "root\WMI" -Class "hpCpsPubGetSetCommand" -Name "hpCpsPubSetCommand" -ArgumentList @("keyboard", [uint32]1, [byte[]]@(3), [uint32]0) -ErrorAction Stop
    $result += "hpCpsPub: ReturnCode=$($r.ReturnCode)`n"
} catch { $result += "hpCpsPub: $($_.Exception.Message.Substring(0,[Math]::Min(80,$_.Exception.Message.Length)))`n" }

# Try HP BIOS keyboard setting
try {
    $bios = Get-WmiObject -Namespace "root\HP\InstrumentedBIOS" -Class "HP_BIOSSetting" -ErrorAction Stop | Where-Object { $_.Name -match "Keyboard" }
    $result += "BIOS keyboard items: $(($bios | Select -ExpandProperty Name) -join ', ')`n"
} catch { $result += "BIOS: $($_.Exception.Message.Substring(0,[Math]::Min(80,$_.Exception.Message.Length)))`n" }

$result | Out-File "C:\Yaariok\schtask_result.txt" -Encoding UTF8
