
import cv2
import time
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector as hd

capture = cv2.VideoCapture(0)  # usa la webcam especificada en el parentesis
capture.set(3,1280)  # ancho
capture.set(4,720)

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # valores default (cntrl click)
mpDraw = mp.solutions.drawing_utils
detector = hd()  # valores por defecto

p_time = 0
c_time = 0

while True:

    success, img = capture.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # manos = detector.findHands(img)
    results = hands.process(imageRGB)  # procesa el frame de la camara y muestra los resultados

    if results.multi_hand_landmarks:
        for handlndmks in results.multi_hand_landmarks:
            for id, lm in enumerate(handlndmks.landmark):
                # cv2.putText(img,str(id), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1,mpDraw.GREEN_COLOR,0)
                # id marca el dedo que se mueva y lm marca la posición en el espacio 3d

                h, w, c = img.shape
                cx, cy = int(lm.x * h), int(lm.y * w)

            mpDraw.draw_landmarks(img,handlndmks,mpHands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time   # display de fps en las imagenes

    cv2.putText(img, str(round(fps)), (90,100), cv2.FONT_HERSHEY_SIMPLEX, 1, mpDraw.GREEN_COLOR, 2)

    cv2.imshow("webcam 0", img)
    cv2.waitKey(1)
