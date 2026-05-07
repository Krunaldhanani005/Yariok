Add-Type -AssemblyName System.Runtime.WindowsRuntime

$null = [Windows.Devices.Lights.LampArray, Windows.Devices.Lights, ContentType=WindowsRuntime]
$null = [Windows.Devices.Enumeration.DeviceInformation, Windows.Devices.Enumeration, ContentType=WindowsRuntime]

$getAwaiter = [System.Runtime.CompilerServices.TaskAwaiter]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { 
    $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.IsGenericMethod 
}

function AwaitIAsync($op) {
    $t = $op.GetType()
    $iface = $t.GetInterface("IAsyncOperation``1")
    if ($iface) {
        $resultType = $iface.GetGenericArguments()[0]
        $method = ($asTask | Where-Object { $_.GetGenericArguments().Count -eq 1 } | Select-Object -First 1)
        if ($method) {
            $task = $method.MakeGenericMethod($resultType).Invoke($null, @($op))
            $task.Wait()
            return $task.Result
        }
    }
    $op.GetResults()
}

try {
    $selector = [Windows.Devices.Lights.LampArray]::GetDeviceSelector()
    $findOp = [Windows.Devices.Enumeration.DeviceInformation]::FindAllAsync($selector)
    $devices = AwaitIAsync $findOp
    "Devices: $($devices.Count)" | Out-File "C:\Yaariok\lamp_admin_test.txt"
} catch {
    "Error: $_" | Out-File "C:\Yaariok\lamp_admin_test.txt"
}
