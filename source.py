import cv2
import numpy as np
import requests
from pytube import YouTube


# function to download the video from YouTube
def download_video(url):
    video = requests.get(url)
    with open("video.mp4", "wb") as f:
        f.write(video.content)

def Download(link):
    youtubeObject = YouTube(link)
    # youtubeObject = youtubeObject.streams.get_highest_resolution()
    youtubeObject = youtubeObject.streams.filter(progressive=True, file_extension='mp4').get_lowest_resolution()
    video_filename = youtubeObject.default_filename
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully for {}".format(video_filename))

    return video_filename

# function to perform object detection on the video
# def object_detection(video_path, model_proto="assets/MobileNetSSD_deploy.prototxt", model_weights="assets/MobileNetSSD_deploy.caffemodel"):
#     # load the pre-trained model
#     model = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
#     # open the video file
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
#         model.setInput(blob)
#         detections = model.forward()
#         for i in np.arange(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.2:
#                 idx = int(detections[0, 0, i, 1])
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
#         cv2.imshow("Output", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


def object_detection(video_path, model_proto="assets/MobileNetSSD_deploy.prototxt", model_weights="assets/MobileNetSSD_deploy.caffemodel", threshold=0.2):
    # load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
    # open the video file
    cap = cv2.VideoCapture(video_path)

    while True:
            # Capture frame-by-frame
        ret, frame = cap.read()
        frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        #Size of frame resize (300x300)
        cols = frame_resized.shape[1] 
        rows = frame_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > threshold: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)
                
                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/300.0  
                widthFactor = frame.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                            (0, 255, 0))

                # Draw label and confidence of prediction in frame resized
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    print(label) #print class and confidence

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) >= 0:  # Break with ESC 
            break

# main function to run the program
def main(url):
    # video_filepath = Download(url)
    video_filepath = "Tesla Self Driving vs Everyday Roads!.mp4"
    object_detection(video_filepath)

# example usage
if __name__ == "__main__":
    url_link = "https://www.youtube.com/watch?v=9nF0K2nJ7N8"
    main(url_link)