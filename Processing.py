

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
    return mask

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
        mask2 = masksum / 255
        #mask1, M, Minv = perspective_transform(mask2, src_pts, dst_pts)
    return mask2
