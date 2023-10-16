import numpy as np
import cv2
import numpy as np
import random
from utils import draw_arousal_valence

if __name__ == "__main__":
    cap = cv2.VideoCapture('E:/mer_data/陶老师的数据/2021-09-16/痴呆/18龚芳秀/05.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        size = (800, 600)
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        expr_name = "happy"
        label = "expr_name: {}".format(expr_name)
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        arousal, valence = random.random(),random.random()
        draw_arousal_valence(frame,arousal,valence)
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()