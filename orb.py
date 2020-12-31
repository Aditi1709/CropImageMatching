import cv2
# import matplotlib.python as plt
import numpy as np 

image1 = cv2.imread("dress.jpg")

training_image = image1
#test_image = image1
test_image = cv2.imread("dresscrop.jpg")
#num_rows, num_cols = test_image.shape[:2]

#rotation_matrix = cv2.getRotationMatrix2D((num_cols/2 , num_rows/2),30, 1)
#test_image = cv2.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))
# test_image = cv2.blur(test_image,(3,3))

training_gray = cv2.cvtColor(training_image,cv2.COLOR_RGB2GRAY)
test_gray = cv2.cvtColor(test_image,cv2.COLOR_RGB2GRAY)

# cv2.imshow("Training image",training_image)

# cv2.imshow("Test image",test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# fx, plots = plt.subplots(1,2,figsize=(20,10))

# plots[0].set_title("Training image")
# plots[0].imshow(training_image)

# plots[1].set_title("Testing image")
# plots[1].imshow(test_image)

orb = cv2.ORB_create()

train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

keypoints = np.copy(training_image)
print(len(keypoints))
cv2.drawKeypoints(training_image,train_keypoints,keypoints,color = (0,0,255))

# cv2.imshow("Training image",keypoints)

# # cv2.imshow("Test image",test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(train_descriptor, test_descriptor)

#matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(training_image,train_keypoints,test_image,test_keypoints,matches[:200],test_gray,flags=2)

cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
