import cv2
import numpy as np

original_image = cv2.imread('test/test001.png')

def freely_select_from_image(original_image):
    """
        Input:
            original_image
        Output:
            masked_image
        Description:
            Freely remove any area from image.
        """
    drawing = False

    def mouse_action(event, former_x, former_y, flags, param):
        global drawing, current_former_x, current_former_y, count
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_former_x,current_former_y=former_x,former_y

        if event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(original_image,(current_former_x,current_former_y),(former_x,former_y),(0,0,0),5)
                current_former_x = former_x
                current_former_y = former_y

        if event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mouse_action)

    while True:
        cv2.imshow("image", original_image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    return original_image

input_image = freely_select_from_image(original_image)

# Take output with pre-trained network
