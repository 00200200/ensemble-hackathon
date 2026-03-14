import cv2
from image_conversion import clean_single_image

img = cv2.imread(r"C:\Users\rondo\Desktop\ensemble\task4\src\data\test\ecg_test_0012.png")
cleaned, cropped, grid_mask, debug = clean_single_image(img)

cv2.imwrite("cleaned.png", cleaned)
cv2.imwrite("cropped.png", cropped)