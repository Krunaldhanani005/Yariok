"""Quick test — verifies the IMOU camera RTSP stream is reachable."""
import cv2

RTSP_URL = "rtsp://Test:Nanta@123@192.168.29.118:554/1/1?transmode=unicast&profile=vam"

print("Connecting to camera...")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("FAILED — Could not connect to camera.")
    print("Check: camera is on, same WiFi, credentials are correct.")
else:
    print("SUCCESS — Camera connected!")
    print(f"Resolution: {int(cap.get(3))} x {int(cap.get(4))}")
    print("Press Q to quit the preview window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost.")
            break
        cv2.imshow("IMOU Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
