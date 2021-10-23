

classes = open('D:/Do an 3/Souce Code/yolov4.txt').read().strip().split('\n')
np.random.seed(42)
weights = "D:/Do an 3/Souce Code/yolov4.weights"
config = "D:/Do an 3/Souce Code/yolov4.cfg"
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
net = cv2.dnn.readNet(weights, config)
scale = 1/255
conf_threshold = 0.5
nms_threshold = 0.4
def vehicle(image):
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    r = blob[0, 0, :, :]
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * right_px)
                center_y = int(detection[1] * bottom_px)
                w = int(detection[2] * right_px)
                h = int(detection[3] * bottom_px)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices,boxes, class_ids, confidences