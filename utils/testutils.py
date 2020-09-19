import cv2
import numpy as np
from scripts.config import Config
from matplotlib import pyplot as plt

cfg = Config()
drawing = False

def freely_select_from_image(org_img):
    """
        Input:
            original_image
        Output:
            masked_image
        Description:
            Freely remove any area from image.
        """
    img = org_img.copy()
    mask = np.empty_like(img)

    def mouse_action(event, former_x, former_y, flags, param):
        global current_former_x, current_former_y, count, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_former_x,current_former_y=former_x,former_y

        if event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255), cfg.freely_select_mask_size)
                current_former_x = former_x
                current_former_y = former_y

        if event == cv2.EVENT_LBUTTONUP:
            drawing = False

        if (event == cv2.EVENT_MOUSEWHEEL):

            if (flags>0):
                cfg.freely_select_mask_size += 1;
            else:
                cfg.freely_select_mask_size -= 1;


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mouse_action)

    while True:
        cv2.imshow("image", img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    mask = cv2.subtract(img[:,:,0], org_img[:,:,0]) # Renkler karisiyor ama sadece maske lazim oldugu icin onemseme su an
    ret,mask = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    return img, mask

def select_by_edge(org_img):
    img = org_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def mouse_action(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(0,len(contours)):
                r=cv2.pointPolygonTest(contours[i], (x,y),False)
                if r > 0:
                    cv2.floodFill(img_gray, contours[i], (0, 0), 150)

        if event == cv2.EVENT_RBUTTONDOWN:
            pass

    v = np.median(img_gray)

    #---- Apply automatic Canny edge detection using the computed median----
    cfg.thresh1 = int(max(0, (1.0 - 0.33) * v))
    cfg.thresh2 = int(min(255, (1.0 + 0.33) * v))

    while True:
        edged = cv2.Canny(img_gray, cfg.thresh1, cfg.thresh2)
        edged_not = cv2.bitwise_not(edged)
        ret, thresh = cv2.threshold(edged_not, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(edged, contours, -1, (0,255,0), 3)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',mouse_action)
        cv2.imshow("image", edged_not)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord('h'):
            cfg.thresh1 += 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord('b'):
            cfg.thresh1 -= 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord('j'):
            cfg.thresh2 += 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord('n'):
            cfg.thresh2 -= 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")

    return org_img
