import cv2
import numpy as np
cap = cv2.VideoCapture('D:/Do an 3/Test/Video/project_video.mp4')
while (cap.isOpened()):
    ret, image = cap.read()
    if image is None:
        break

    imgA = np.ones((720, 1280))#, np.uint8)

    indices, boxes, class_ids, confidences = vehicle(image)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        X = x + 0.5 * w
        Y = y + 0.5 * h
        color = color = [int(c) for c in colors[class_ids[i]]]
        label = str(classes[class_ids[i]])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, label + "{:.2f}".format(confidences[i]), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        imgA[y - 10:y + h + 20, x - 30:x + w + 30] = 0
   
    mask = process(image)
    mask2 = mask / 255
    mask1, M, Minv = perspective_transform(mask2, src_pts, dst_pts)
    B = cv2.countNonZero(mask1)
    if (B < 4750):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        masksum = cv2.bitwise_or(mask, sxbinary * 255)
        mask2 = cv2.bitwise_and(mask, imgA * 255)
        mask2 = masksum / 255
    A=False
    if not left_line.detected or not right_line.detected:
        left_fitx, right_fitx, ploty,left_fit,right_fit,left_lane_inds,right_lane_inds, A = find_line(mask1)

        left_line.best_fit= left_fit
        right_line.best_fit= right_fit

    else:

        left_fitx, right_fitx, ploty,left_fit,right_fit, left_lane_inds,right_lane_inds=previouslane(mask1,left_line.best_fit,right_line.best_fit)
    left_line.add_best_fit(left_fit, left_lane_inds)
    right_line.add_best_fit(right_fit, right_lane_inds)


    if left_line.best_fit is not None and right_line.best_fit is not None:
        result = draw_on_original(image, left_line.best_fit, right_line.best_fit, ploty, Minv)
    #cv2.putText(result, "{:.2f}".format(end), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),1)

    if (A):
        left_line.detected= False
        right_line.detected= False
        #left_line.best_fit = None
        #right_line.best_fit = None
    if cv2.waitKey(33) >= 0:
        break
cap.release()
out.release()
cv2.destroyAllWindows()