"""
Keyboard RGB control for HP Victus via Windows LampArray API.
Runs as SYSTEM via scheduled task (no UAC needed after setup).
Usage: python kb_control.py on|off|red|blue|white
"""
import sys, asyncio

MODE = (sys.argv[1] if len(sys.argv) > 1 else "on").lower()

COLORS = {
    "on":    (255, 255, 255, 255),   # white, full brightness
    "off":   (255, 0,   0,   0),     # off (alpha=255, rgb=000 → all off)
    "red":   (255, 255, 0,   0),
    "green": (255, 0,   200, 0),
    "blue":  (255, 0,   80,  255),
    "white": (255, 255, 255, 255),
    "warm":  (255, 255, 120, 20),
}

a, r, g, b = COLORS.get(MODE, COLORS["on"])

async def main():
    import winrt.windows.devices.lights as wdl
    import winrt.windows.devices.enumeration as wde
    import winrt.windows.ui as wu

    sel = wdl.LampArray.get_device_selector()
    devices = await wde.DeviceInformation.find_all_async_aqs_filter(sel)
    if not devices or len(devices) == 0:
        print("No LampArray devices found"); return

    lamp_array = await wdl.LampArray.from_id_async(devices[0].id)
    if not lamp_array:
        print("LampArray connection failed"); return

    color = wu.Color(a, r, g, b)
    lamp_array.set_color(color)
    for i in range(lamp_array.lamp_count):
        lamp_array.set_color_for_index(i, color)

    print(f"OK lamp_count={lamp_array.lamp_count} mode={MODE} available={lamp_array.is_available}")

asyncio.run(main())
