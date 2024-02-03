import cv2
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr

# Specify the paths to the image, config file, weights, and classes file
config_path = r'C:\Users\ADMIN\Desktop\python_scripts\yolov3.cfg'
weights_path = r'C:\Users\ADMIN\Desktop\python_scripts\yolov3.weights'
classes_path = r'C:\Users\ADMIN\Desktop\python_scripts\yolov3.txt'

# Load the classes and colors outside the loop
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize the YOLO network outside the loop
net = cv2.dnn.readNet(weights_path, config_path)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set the rate (adjust this value to control the speed)
#engine.setProperty('rate', 1)

cap = cv2.VideoCapture(0)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    text = f"there is a {label} in front of you"      #spech engine parameters
    print(text)
    engine.say(text)
    engine.runAndWait()
    #time.sleep()



def handle_detected_objects(boxes, class_ids, confidences):
    detected_objects = []
    for i in range(len(boxes)):
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        detected_objects.append(classes[class_ids[i]])
    return detected_objects

def respond_to_objects(detected_objects):
    if detected_objects:
        engine.say(f"The detected objects are: {', '.join(detected_objects)}")
        engine.runAndWait()
    else:
        engine.say("Object not recognized.")
        engine.runAndWait()

def speech_recognition_thread():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening for the command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            if "what is in the frame" in command:
                respond_to_objects(detected_objects)
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Speech recognition request failed: {e}")

# Start the thread for speech recognition
#speech_thread = threading.Thread(target=speech_recognition_thread)
#speech_thread.start()

while True:
    ret, frame = cap.read()
    #if ret:
        #cv2.imshow("Object Detection",frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

    Width = 1280
    Height = 720
    image = cv2.resize(frame, (1280, 720))
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    try:
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        detected_objects = handle_detected_objects(boxes, class_ids, confidences)


    except Exception as e:
        # Handle exception (no objects detected)
        pass

    cv2.imshow("object detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
