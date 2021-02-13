import cv2
import matplotlib.pyplot as plt


img = cv2.imread('data/train_masks/1.bmp', cv2.IMREAD_GRAYSCALE)
print(img)

img = cv2.imread('saved_images/pred_0.png', cv2.IMREAD_GRAYSCALE)
print(img)

plt.imshow(img)
plt.show()


img = cv2.imread('dataOld/train_masks/1e6f48393e17_01_mask.gif', cv2.IMREAD_GRAYSCALE)
print(img)

# plt.imshow(img)
# plt.show()



