import cv2
import numpy as np

#---------- Image Info ---------- #

# The target image for detection
image = cv2.imread("TestImage.jpg")

#---------- Getting the Data ---------- #

# Pull the data from the tensor produced by the model in the .onnx file
def extractData(tensor, thresh):

    return_arr = []

    bounding_boxes = []
    confidences = []
    class_IDs = []

    # Manipulate the data so that it makes sense (I have no idea why this is done, but the documentation does it)
    transposed_tensor = np.array([cv2.transpose(tensor[0])])
    readable_data = transposed_tensor[0]

    for data in readable_data:

        # All values after the first four indexes are confidence scores
        class_scores = data[4:]

        # The first four indexes are the bounding box information
        x, y, w, h = data[:4]
        
        # The confidence that the identification is correct
        confidence = np.max(class_scores)

        # The ID of the identification
        class_ID = np.argmax(class_scores)

        # Filter out unconfident identifications
        if confidence >= thresh:

            bounding_boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_IDs.append(class_ID)

    # Due to the way object detection works, some objects will be counted multiple times. 
    # Remove the multiples
    result_boxes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, thresh, 0.5)

    for valid_boxes in result_boxes:

        temp = []

        temp += bounding_boxes[valid_boxes]
        temp.append(confidences[valid_boxes])
        temp.append(class_IDs[valid_boxes])

        return_arr.append(temp)
            
    return return_arr

#---------- Feeding in Images ---------- #

# Set up the model from the .onnx file
net = cv2.dnn.readNetFromONNX("v8.onnx")

# Alter the image to match the requirements by the model
# Things like aspect ratio and colour are considered
blob = cv2.dnn.blobFromImage(image, 1/255 , (640, 640), swapRB=True, mean=(0,0,0), crop= False)

# Recieve the altered image
net.setInput(blob)

# Pull the output
outputs =  net.forward()

data = extractData(outputs, 0.25)

#---------- Showing the Bounding Boxes ---------- #

for detection in data:

    print(detection)

    x, y, w, h = map(int, detection[:4])

    x1 = int(x - (w / 2))
    y1 = int(y - (h / 2))

    x2 = int(x + (w / 2))
    y2 = int(y + (h / 2))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Test", image)

cv2.waitKey(0)

    

    

