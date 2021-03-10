import numpy as np
import cv2
from time import time
from time import sleep
import threading
from matplotlib import pyplot as plt
import pickledb
import serial
plt.close('all')


def read_db():
    global w, h, x, y, min_width, max_width, min_height, max_height, y_min_lim, y_max_lim
    global pattern_pass_time, cut_delay
    print('Reading configurations...')
    try:
        config = pickledb.load('./config.db', False)
    except:
        print('Error reading configurations \n')
    else:
        print('Reading configurations successful! \n')
        w = config.get('width')
        min_width = config.get('min_width')
        max_width = config.get('max_width')
        h = config.get('height')
        min_height = config.get('min_height')
        max_height = config.get('max_height')
        x = config.get('start_x')
        y = config.get('start_y')
        y_min_lim = config.get('y_min_lim')
        y_max_lim = config.get('y_max_lim')
        pattern_pass_time = config.get('pattern_pass_time') / 100
        cut_delay = config.get('cut_delay') / 100


def prediction_timer():
    global newcut
    global list_time, list_dist, cut_delay
    print('Prediction timer ended \n')
    newcut = True

    if len(list_time) > 1:
        list_rel_time = [val-list_time[0] for val in list_time]
        p = np.polyfit(list_dist, list_rel_time, deg=1)
        time_to_0 = p[0] * 0 + p[1]
        wait_time = time_to_0 + list_time[0] - time() + cut_delay
        t = threading.Timer(wait_time, relay_timer)
        t.start()
        print('Relay timer started \n')
    else:
        print('Not enough values for line fit \n')

    list_time = []
    list_dist = []


def relay_timer():
    global ser
    print('Relay timer ended \n')
    print('Writing serial for relay -----------------> \n')
    try:
        ser.write(b'H')
    except:
        print('!!!!!!!   Serial port not working   !!!!!!!')
        try:
            ser = serial.Serial('/dev/ttyACM0')  # open serial port
            print(ser.name)  # check which port was really used
        except:
            print('Serial (USB) port STILL not working. Check connection!')
        else:
            print('Serial (USB) port successfully opened \n')



def db_timer():
    global db_timer_expired
    db_timer_expired = True


print('Opening serial (USB) port \n')
while True:
    try:
        ser = serial.Serial('/dev/ttyACM0')  # open serial port
        print(ser.name)  # check which port was really used
    except:
        print('Serial (USB) port not working. Check connection!')
    else:
        print('Serial (USB) port successfully opened \n')
        break
    sleep(3)
print('Waiting for arduino to stabalize \n')
sleep(2)

cut_cascade = cv2.CascadeClassifier('../haar_trained_model/cascade.xml')
cap = cv2.VideoCapture('./rpi.mp4') 	# Test video
# cap = cv2.VideoCapture(0)			# Capture video from Rpi Camera

newcut = True
list_time = []
list_dist = []
db_timer_expired = True

while cap.isOpened():

    if db_timer_expired is True:
        read_db()
        db_timer_expired = False
        timer_3 = threading.Timer(5, db_timer)
        timer_3.start()

    ret, img_orig = cap.read()
    start_time = time()
    if ret is True:
        time_imgcapture = time()
        img = img_orig[y:y+h, x:x+w]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 53, 37)
        blur = cv2.medianBlur(gaus, 7)
        cuts = cut_cascade.detectMultiScale(gaus, scaleFactor=1.01, minNeighbors=1, minSize=(min_width, min_height),
                                            maxSize=(max_width, max_height))

        loc = []
        locavg = None
        cv2.rectangle(img_orig, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.rectangle(img_orig, (x+int(w/2-min_width/2), y+int(h/2-min_height/2)),
                      (x+int(w/2-min_width/2) + min_width, y+int(h/2-min_height/2) + min_height), (0, 255, 255), 2)
        cv2.rectangle(img_orig, (x + int(w / 2 - max_width / 2), y + int(h / 2 - max_height / 2)),
                      (x + int(w / 2 - max_width / 2) + max_width, y + int(h / 2 - max_height / 2) + max_height),
                      (0, 255, 255), 2)

        for (x1, y1, w1, h1) in cuts:
            cv2.rectangle(img_orig, (x1+x, y1+y), (x1+x + w1, y1+y + h1), (255, 0, 255), 2)
            loc.append(y1+int(h1/2))
        if loc:
            locavg = int(np.average(loc))
            cv2.line(img_orig, (0, locavg), (900, locavg), (255, 0, 255), 2)

        if locavg is not None:
            if y_max_lim > locavg > y_min_lim:
                list_time.append(time_imgcapture)
                list_dist.append(locavg)
                if newcut is True:
                    newcut = False
                    print('Prediction timer started \n')
                    timer1 = threading.Timer(pattern_pass_time, prediction_timer)
                    timer1.start()

        cv2.imshow('img_orig', img_orig)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

ser.close()
