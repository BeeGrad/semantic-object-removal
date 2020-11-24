import cv2
import numpy as np
from scripts.config import Config
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray
from scripts.dataOperations import DataRead
import torch

cfg = Config()
drawing = False


def freely_select_from_image(org_img):
    """
        Input:
            original_image
        Output:
            masked_image: input image for network
            mask: mask that is used on image
            image_gray: gray version if mask masked_image
            edge: edge map of masked_image
        Description:
            Freely remove any area from image.
        """
    img = org_img.copy()
    mask = np.empty_like(img[:,:,0])
    edge = np.empty_like(img[:,:,0])
    image_gray = np.empty_like(img[:,:,0])
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(image_gray, cfg.thresh1, cfg.thresh2)
    # image_gray = rgb2gray(img) # Nedenini bilmiyorum ama ust 2 satir yerine bu 2 satiri kullaninca
    # edge = canny(image_gray, sigma=cfg.SIGMA) # Hata veriyor, edge uzerine line cizerken
    # print(type(edge))

    def mouse_action(event, former_x, former_y, flags, param):
        global current_former_x, current_former_y, count, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_former_x, current_former_y = former_x, former_y

        if event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255), cfg.freely_select_mask_size)
                cv2.line(edge,(current_former_x,current_former_y),(former_x,former_y),(0,0,0), cfg.freely_select_mask_size)
                cv2.line(mask,(current_former_x,current_former_y),(former_x,former_y),(255,255,255), cfg.freely_select_mask_size)
                current_former_x = former_x
                current_former_y = former_y

        if event == cv2.EVENT_LBUTTONUP:
            drawing = False

        if event == cv2.EVENT_MOUSEWHEEL:

            if flags > 0:
                cfg.freely_select_mask_size += 1
            else:
                cfg.freely_select_mask_size -= 1

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_action)

    while True:
        cv2.imshow("image", img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    return img, mask, image_gray, edge


def select_by_edge(org_img):
    """
    Input:
        org_img: unmasked image
    Output:
        masked_image
    Description:
        This function is not workin currently, it returns the org_img
        Should find a way to select by contours.
    """
    img = org_img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def mouse_action(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(0, len(contours)):
                r = cv2.pointPolygonTest(contours[i], (x, y), False)
                if r > 0:
                    cv2.floodFill(img_gray, contours[i], (0, 0), 150)

        if event == cv2.EVENT_RBUTTONDOWN:
            pass

    v = np.median(img_gray)

    # ---- Apply automatic Canny edge detection using the computed median----
    cfg.thresh1 = int(max(0, (1.0 - 0.33) * v))
    cfg.thresh2 = int(min(255, (1.0 + 0.33) * v))

    while True:
        edged = cv2.Canny(img_gray, cfg.thresh1, cfg.thresh2)
        edged_not = cv2.bitwise_not(edged)
        ret, thresh = cv2.threshold(edged_not, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # cv2.drawContours(edged, contours, -1, (0,255,0), 3)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_action)
        cv2.imshow("image", edged_not)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if k == ord("h"):
            cfg.thresh1 += 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord("b"):
            cfg.thresh1 -= 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord("j"):
            cfg.thresh2 += 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")
        if k == ord("n"):
            cfg.thresh2 -= 10
            print(f"thresh1: {cfg.thresh1}, thresh2: {cfg.thresh2}")

    return org_img

def select_by_train_mask(org_img):
    print(org_img.shape)
    data = DataRead(cfg.dataset, cfg.masking_type, cfg.batch_size)
    imgs, gray_images, edges, masks = data.return_inputs(torch.FloatTensor(org_img).unsqueeze(0).permute(0,3,1,2))
    edges = (1 - edges)
    imgs = np.uint8(imgs.squeeze(0).permute(1,2,0).numpy() * 255)
    gray_images = np.uint8(gray_images.squeeze(0).squeeze(0).numpy() * 255)
    edges = np.uint8(edges.squeeze(0).squeeze(0).numpy() * 255)
    masks = np.uint8(masks.squeeze(0).squeeze(0).numpy() * 255)
    return imgs, gray_images, edges, masks
