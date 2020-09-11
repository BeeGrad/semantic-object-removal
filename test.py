import cv2
import numpy as np
from utils.utils import calculate_psnr

original_image = cv2.imread('../foreground-substraction/test/test001.png')

def freely_select_from_image(org_img):
    """
        Input:
            original_image
        Output:
            masked_image
        Description:
            Freely remove any area from image.
        """
    drawing = False
    img = org_img.copy()

    def mouse_action(event, former_x, former_y, flags, param):
        global drawing, current_former_x, current_former_y, count
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_former_x,current_former_y=former_x,former_y

        if event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(0,0,0),5)
                current_former_x = former_x
                current_former_y = former_y

        if event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mouse_action)

    while True:
        cv2.imshow("image", img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    return img

input_image = freely_select_from_image(original_image)
print(calculate_psnr(input_image, original_image))
# Take output with pre-trained network
