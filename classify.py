from keras.models import load_model
import numpy as np
import cv2
import time

BACK_ELIPSOID = (
	np.array([[0.00011817132881515564, -4.268966406578619e-05, -7.86682832867688e-06],
			[-4.2689664065785295e-05, 0.0011804262381059374, -0.0011823199858874702],
			[-7.866828328677854e-06, -0.0011823199858874698, 0.0012785150794466832]]),
	np.array([71.21917986604076, 174.80637325770218, 164.1857072962927]),
	(255, 0, 0)
)

COLOR_ELIPSOIDS = [
	# (A, Centroid, color)
	# GREEN
	(
		np.array([[0.011206694203799569, -0.009831559831242849, -0.0035002854281720817],
				[-0.009831559831242407, 0.01645394536470453, -0.0024024849063659804],
				[-0.0035002854281724447, -0.0024024849063654804, 0.005926940263370816]]),
		np.array([105.55164041508559, 128.0233320477617, 121.5378747063686]),
		(0, 255, 0)
	),
	# RED
	(
		np.array([[0.011372475348557896, -0.012029134442363698, 0.0006636996096530968],
				[-0.012029134442363698, 0.014963868087775336, -0.001678667616806717],
				[0.0006636996096530969, -0.001678667616806717, 0.0023740536589177473]]),
		np.array([110.34203960972923, 115.92990087946372, 181.64333412723155]),
		(0, 0, 255)
	),
	# BLACK
	(
		np.array([[0.021884687549989856, -0.015848017892524765, -0.005661796083790989],
				[-0.015848017892525646, 0.019729837222934817, -0.0030208084058373826],
				[-0.005661796083790165, -0.003020808405838241, 0.008368513079182679]]),
		np.array([109.39374063652724, 113.99399944422471, 119.93235618205823]),
		(255, 0, 0)
),
]


classification_model = load_model("flattened_model.h5")

if classification_model is None:
	# throw exception
	print("Failed to load classification model")
	raise Exception("Failed to load classification model")

def distance_from_ellipse(A, c, points):
	"""
	Return the distance from a point to an ellipse in center form
	(x-C)' A (x-C)
	points is a list of points where each point is a list of 3 values.
	"""
	return np.einsum('ij,ij->i', np.einsum('ij,jk->ik', points-c, A), points-c)

def flattenImage(img: cv2.Mat) -> tuple[cv2.Mat, int]:

	debug = [img]

	# Tile Back Check:
	pixels_matching_back = distance_from_ellipse(BACK_ELIPSOID[0], BACK_ELIPSOID[1], img.reshape(-1, 3)) < 1
	if np.count_nonzero(pixels_matching_back) >= img.shape[0] * img.shape[1] * 0.2:
		cv2.imwrite("Predictions/back_" + str(np.count_nonzero(pixels_matching_back)) + ".png", img)
		return (img, -2)

	rgb_planes = cv2.split(img)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
		dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
		bg_img = cv2.medianBlur(dilated_img, 33)
		diff_img = 255 - cv2.absdiff(plane, bg_img)
		norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		result_planes.append(diff_img)
		result_norm_planes.append(norm_img)
		
	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)

	debug.append(result_norm)

	norm_gray = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(norm_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	mask = thresh

	debug.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

	contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 0.3 * img.shape[0] * img.shape[1] and cv2.contourArea(cnt) > 0.001 * img.shape[0] * img.shape[1]]

	# filter all contours that touch the edge of the image
	contours = [cnt for cnt in contours if not np.any(cnt[:, 0, 0] == 0) and not np.any(cnt[:, 0, 0] == img.shape[1] - 1) and not np.any(cnt[:, 0, 1] == 0) and not np.any(cnt[:, 0, 1] == img.shape[0] - 1)]

	contoursImage = np.zeros(img.shape[:2], np.uint8)
	cv2.drawContours(contoursImage, contours, -1, 255, -1)
	debug.append(cv2.cvtColor(contoursImage, cv2.COLOR_GRAY2BGR))

	if len(contours) == 0:
		return (img, -1)

	result = np.zeros(img.shape, np.uint8)
	result_avg = np.zeros(img.shape, np.uint8)
	contourColors = []
	for contour in contours:
		# get the average color of the pixels in the contour
		test = np.zeros(img.shape[:2], np.uint8)
		cv2.drawContours(test, [contour], -1, 255, -1)

		avgColor_old = np.mean(result_norm[contour[:, 0, 1], contour[:, 0, 0]], axis=0)

		contourMask = np.zeros(img.shape[:2], np.uint8)
		cv2.drawContours(contourMask, [contour], -1, 255, -1)
		contourMask = cv2.bitwise_and(contourMask, mask)
		avgColor = np.mean(result_norm[contourMask == 255], axis=0)


		# print (avgColor_old, avgColor)

		contourColors.append(avgColor)

		cv2.drawContours(result_avg, [contour], -1, avgColor, -1)

	debug.append(result_avg)


	distances = [
		distance_from_ellipse(COLOR_ELIPSOIDS[0][0], COLOR_ELIPSOIDS[0][1], contourColors),
		distance_from_ellipse(COLOR_ELIPSOIDS[1][0], COLOR_ELIPSOIDS[1][1], contourColors),
		distance_from_ellipse(COLOR_ELIPSOIDS[2][0], COLOR_ELIPSOIDS[2][1], contourColors),
	]

	# sort distances, maintaining indices
	distances = np.array(distances)
	distances = distances.argsort(axis=0)[0]
	contourColors = [COLOR_ELIPSOIDS[i][2] for i in distances]


	for i in range(len(contours)):
		cv2.drawContours(result, [contours[i]], -1, contourColors[i], -1)

	result_avg = cv2.bitwise_and(result_avg, result_avg, mask=mask)
	result = cv2.bitwise_and(result, result, mask=mask)

	avgResultTotalColor = np.mean(result_avg, axis=(0, 1))

	result_int = np.argmax(avgResultTotalColor)

	debug.append(result)

	debugResults = np.hstack(debug)
	cv2.imwrite("Predictions/DebugResults" + str(time.time()) + "_" + str(result_int) + ".png", debugResults)

	return (result, result_int)




def predict_classes(originalImage, bboxes, masks):

	cv2.imwrite("Predictions/OriginalImage.png", originalImage)

	predictions = []
	confidences = []

	for i in range(len(bboxes)):
		bbox = bboxes[i]
		mask = masks[i]
		# Crop image to bounding box
		x1, y1, x2, y2 = bbox
		croppedImage = originalImage[y1:y2, x1:x2]

		mask = mask.reshape((y2 - y1, x2 - x1))

		maskedImage = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
		maskedImage[mask == 1] = croppedImage[mask == 1]

		# Resize image to 224x224
		maskedImage = cv2.resize(maskedImage, (224, 224), interpolation=cv2.INTER_AREA)

		(flattenedImage, flattenCode) = flattenImage(maskedImage)

		if flattenCode == -1:
			predictions.append(36)
			confidences.append(1)
			continue
		elif flattenCode == -2:
			predictions.append(37)
			confidences.append(1)
			continue

		img_arr = np.asarray(flattenedImage, dtype=np.float32).reshape(1, 224, 224, 3)

		# Preform normalization
		normalizedImage = (img_arr / 127.0) - 1

		# Preform classification

		prediction = classification_model.predict(normalizedImage)
		predictionIndex = int(np.argmax(prediction))
		predictionConfidence = prediction[0][predictionIndex]

		cv2.imwrite("Predictions/CroppedImage" + str(i) + "_" + str(predictionIndex) + ".png", maskedImage)

		if predictionIndex >= 12:
			if predictionIndex <= 14:
				# 12, 13, 14 (red fives)
				if flattenCode == 2:
					predictionIndex = predictionIndex + 3
			else:
				# offest skipping 15, 16, 17 (red tens)
				predictionIndex = predictionIndex + 3

				if predictionIndex == 31 and flattenCode == 2:
					# 31 (red dragon)
					predictionIndex = predictionIndex + 1

		predictions.append(predictionIndex)
		confidences.append(float(predictionConfidence))

		# Preform classification

	return predictions, confidences