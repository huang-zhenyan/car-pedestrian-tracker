import cv2

#image
img_file = "car image.jpg"
#video = cv2.VideoCapture("Teslas Avoids Accidents.mp4")
video = cv2.VideoCapture("Tesla Pedestrians.mp4")

#creat opencv image
#img = cv2.imread(img_file)

#convert to grayscale (needed for Haar Cascade)
#black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#pre-trained car classifer
car_tracker_file = "haarcascade_cars.xml"
pedestrian_tracker_file = "haarcascade_fullbody.xml"


#create classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#run forever until car stops
while True:
    #read the current frame
    (read_successful, frame) = video.read()

    #safe coding
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #display the image
    cv2.imshow("car tracker", frame)

    #do not autoclose until key is pressed
    key = cv2.waitKey(1)

    #stop if Q is pressed
    if key==81 or key==113:
        break

#release video capture
video.release()






print("code completed")