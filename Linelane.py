import cv2
import numpy as np
import time
import glob
import matplotlib.pyplot as plt

(bottom_px, right_px) =(720,1280)

pts = np.array([[210,bottom_px],[565,450],[700,450], [1110, bottom_px]], np.int32)
cap = cv2.VideoCapture('D:/Do an 3/Test/Video/project_video.mp4')

src_pts = pts.astype(np.float32)
dst_pts = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)

def perspective_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def process(image):
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    Whitemin = np.uint8([0, 200, 0])
    Whitemax = np.uint8([255, 255, 225])
    whitemask = cv2.inRange(image2, Whitemin, Whitemax)
    #Yemin = np.uint8([10, 100, 80])
    #Yemax = np.uint8([40, 255, 255])
    Yemin = np.uint8([10, 60, 80])
    Yemax = np.uint8([160, 255, 255])
    yemask = cv2.inRange(image2, Yemin, Yemax)
    mask = cv2.bitwise_or(whitemask, yemask)
    #cv2.imshow("Y1", whitemask)
    mask1 = cv2.bitwise_and(image, image, mask=mask)
    #img = cv2.polylines(mask1, [pts], True, (0, 255, 0), 3)
    #cv2.imshow('Q',mask1)
    return mask
def find_line(binary_warped):
    haff= binary_warped[binary_warped.shape[0] // 2:, :]
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    #plt.plot(histogram)
    #plt.imshow(binary_warped)
    #plt.show()
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    nwindows = 9

    window_height = np.int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        #cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:

            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    A=False
    left_x_mean = np.mean(leftx, axis=0)
    right_x_mean = np.mean(rightx, axis=0)
    lane_width = np.subtract(right_x_mean, left_x_mean)
    if len(leftx) <= 1000 or len(rightx) <= 1000 or lane_width < 300 or lane_width > 800 :
        A=True
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if (len(leftx)!=0):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    else:
        left_fit=None
        left_fitx = 1 * ploty ** 2 + 1 * ploty
    if (len(rightx)!=0):
        right_fit = np.polyfit(righty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    else:
        right_fit=None
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return  left_fitx, right_fitx, ploty,left_fit,right_fit,left_lane_inds ,right_lane_inds, A

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = np.zeros(720)
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        self.line_base_pos = np.zeros(1)
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.pre=False
        # smoothen the n frames
        self.smoothen_nframes = 10
        # first frame
        self.first_frame = True

    def add_best_fit(self, lane_fit, lane_inds):
        if lane_fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(lane_fit - self.best_fit)
                if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit)> 0:

                    #if M<=2:
                    self.detected = False
                    #print('AA')
                else:
                    self.detected = True
                    self.px_count = np.count_nonzero(lane_inds)
                    self.current_fit.append(lane_fit)
                    if len(self.current_fit) > 5:
                        self.current_fit = self.current_fit[len(self.current_fit) - 5:]
                    self.best_fit = lane_fit
            else:
                self.best_fit = lane_fit
                #self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False
            if len(self.current_fit) > 2:
                self.current_fit = self.current_fit[:len(self.current_fit) - 1]
                self.best_fit = np.average(self.current_fit, axis=0)
            else:
                self.current_fit = self.current_fit[:len(self.current_fit)]
                self.best_fit = np.average(self.current_fit, axis=0)

def previouslane(binary_warped,prev_left_fit,prev_right_fit):
    #color_warp = np.zeros_like(binary_warped).astype(np.uint8)

    #out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (prev_left_fit[0] * (nonzeroy ** 2) + prev_left_fit[1] * nonzeroy +
                                   prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0] * (nonzeroy ** 2) +
                                                                              prev_left_fit[1] * nonzeroy +
                                                                              prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (prev_right_fit[0] * (nonzeroy ** 2) + prev_right_fit[1] * nonzeroy +
                                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0] * (nonzeroy ** 2) +
                                                                                prev_right_fit[1] * nonzeroy +
                                                                                prev_right_fit[2] + margin))).nonzero()[0]

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    #print('len='+str(len(righty)))
    if len(lefty) == 0:
        left_fit = prev_left_fit
    if len(righty) == 0:
        right_fit = prev_right_fit
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    # left_line.current_fit = left_fit
    # right_line.current_fit = right_fit

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    return left_fitx, right_fitx, ploty,left_fit,right_fit,left_lane_inds,right_lane_inds
left_line = Line()
right_line = Line()
def draw_on_original(undist, left_fit, right_fit, ploty, Minv):
    color_warp = np.zeros_like(undist).astype(np.uint8)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)
    return result

def findmask(image):
    mask = process(image)
    mask2 = mask / 255
    mask1, M, Minv = perspective_transform(mask2, src_pts, dst_pts)
    B = cv2.countNonZero(mask1)
    #im = cv2.polylines(image, [pts], True, (0, 255, 255), 3)
    #print(B)
    #cv2.imshow("Ma", im)
    #cv2.waitKey()
    if (B < 4750):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        masksum = cv2.bitwise_or(mask, sxbinary * 255)
        # masksum = cv2.bitwise_and(masksum,img*255)
        # cv2.imwrite("D:/Do an 3/AA1.jpg", masksum)
        #cv2.imshow("Ma1", masksum)
        mask2 = masksum / 255
        #mask1, M, Minv = perspective_transform(mask2, src_pts, dst_pts)
    return mask2

FILE_OUTPUT = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (1280, 720))

classes = open('D:/Do an 3/Souce Code/yolov3.txt').read().strip().split('\n')
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
#frame_id=0
while (cap.isOpened()):
    ret, image = cap.read()
    if image is None:
        break
    sta=time.time()
    #mask2 = cv2.bitwise_and(mask2*255, imgA*255)
    #mask1, M, Minv = perspective_transform(mask2/255, src_pts, dst_pts)
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
        # masksum = cv2.bitwise_and(masksum,img*255)
        # cv2.imwrite("D:/Do an 3/AA1.jpg", masksum)
        # cv2.imshow("Ma1", masksum)
        mask2 = cv2.bitwise_and(mask, imgA * 255)
        mask2 = masksum / 255
    A=False
    #pre=False
    if not left_line.detected or not right_line.detected:
        left_fitx, right_fitx, ploty,left_fit,right_fit,left_lane_inds,right_lane_inds, A = find_line(mask1)
        #left_line.pre= True
        #right_line.pre=True
        #left_line.detected=None
        #right_line.detected= None
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
    out.write(result)
    cv2.imshow('mau', result)
    if cv2.waitKey(33) >= 0:
        break
cap.release()
out.release()
cv2.destroyAllWindows()