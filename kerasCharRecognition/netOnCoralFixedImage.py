import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import time

from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import pycoral.utils.edgetpu

from pypylon import pylon

class BaslerCam():
	#inits a Basler camera object
	def __init__(self):
		self.img = pylon.PylonImage()
		self.tlf = pylon.TlFactory.GetInstance()
		self.cam = pylon.InstantCamera(self.tlf.CreateFirstDevice())

	#grabs a single image, converts it and returns the result
	def grabImage(self):
		self.cam.Open()
		self.cam.StartGrabbing()

		with self.cam.RetrieveResult(2000) as result:

			# Calling AttachGrabResultBuffer creates another reference to the
			# grab result buffer. This prevents the buffer's reuse for grabbing.
			self.img.AttachGrabResultBuffer(result)

			self.img.Release()

			return self.convertImage(result)


	# converts the picture of the camera to opencv bgr format and returns it
	def convertImage(self, result):
		converter = pylon.ImageFormatConverter()
		converter.OutputPixelFormat = pylon.PixelType_Mono8
		converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

		#print(result)
		imageConverted = converter.Convert(result)
		imageArray = imageConverted.GetArray()

		return imageArray

	#save an Image as bmp
	def saveImage(self, result):
		#filename = "saved_pypylon_img.png"
		#result.Save(pylon.ImageFileFormat_Bmp, filename)

		imgToSave = Image.fromarray(result)
		imgToSave.save("saved_pypylon_img.bmp")


def main():

	charList = ['0','1','2','3','4','5','6','7','8','9',
	'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
	'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	#rudimentäre Ausschriften vom Coral einschalten
	pycoral.utils.edgetpu.set_verbosity(1)

	#Auflistung aller Corals
	print(pycoral.utils.edgetpu.list_edge_tpus())

	#Netz einem Coral zuweisen
	interpreter = make_interpreter("tfKerasChars10Epochs.tflite", device=':0')

	#Netz "initialisieren"
	interpreter.allocate_tensors()

#	#Input Daten von der Kamera holen und für das Netz vorbereiten
#	cam = BaslerCam()
#	imageArray = cam.grabImage()
#	print("Bildauflösung", imageArray.shape)


	#Festes Bild laden und vorverarbeiten
	image  = cv2.imread("testBild.jpg")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	threshInv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 20)

#	plt.imsave("gray.png", gray, cmap='gray')
#	plt.imsave("threshInv.png", threshInv, cmap='gray')

	#Rechtecke ermitteln
	contours, _ = cv2.findContours(threshInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnt = contours[0]

	rects = []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		rects.append((x,y,w,h))
	rects.reverse()

	words = []
	letters = []

	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 20)
#	plt.imsave("thresh.png", thresh, cmap='gray')

	alt_y = -1

	for r in rects:
		x,y,w,h = r
		if (w >= 56 and w <= 60) and (h >= 53 and h <= 67):
			#jump to next word
			if y > alt_y and alt_y != -1:
				words.append(letters)
				letters = []

			if False: #uninverted image
				#get single letter from big image
				crop_img = thesh[y:y+h, x:x+w]

				#only take letters which are not blank
				if crop_img.sum() < 753032:
					letters.append(crop_img)
					alt_y = y
			else: #inverted image
				#get single letter from big image
				crop_img = threshInv[y:y+h, x:x+w]

				#only take letters which are not blank
				if crop_img.sum() > 56610:
					letters.append(crop_img)
					alt_y = y

	#append last Word, as there is no linebreak
	words.append(letters)

	#interate trough all letters of all words to the separated chars
	for word in words:
		str = ''
		cnt = 0
		for l in word:
			imageArray = l

			#Bild von Eingangsgöße auf 28x28 skalieren
			resImage = cv2.resize(imageArray, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
			#Daten von uint8 auf fp32 ändern und auf Werte von 0 bis 1 bringen
			normImage = resImage.astype(np.float32)[:]/255
			#print(type(normImage[1,1]))
			#print(normImage)
	#		plt.imsave("{0}.png".format(cnt), normImage, cmap='gray')
	#		cnt += 1

	#		print("Form der Input Daten", normImage.shape)

			#Input für das Netz setzen
			common.set_input(interpreter, normImage)
			#Unterschiedliche Infos zum Input Tensor erfassen
			#print("Daten des Input Tensors des Netzes", common.input_tensor(interpreter))
			#print("Input Details", common.input_details(interpreter, "shape"))
			interpreter_infos = interpreter.get_input_details()[0]
	#		for _ in interpreter_infos:
	#			print(_,  interpreter_infos[_])

			#NN laufen lassen
			startZeit= time.process_time_ns()
	#		pycoral.utils.edgetpu.run_inference(interpreter, normImage)
			interpreter.invoke()
	#		print("Klassifizierungszeit in µs {0}".format((time.process_time_ns() - startZeit)/1000))

			#Ergebnis vom NN abholen
			output = common.output_tensor(interpreter, 0)
			#print("Output Tensor", output)

			#Ergebnis aufbereiten
	#		print("Zeichen unmapped", np.argmax(output[0]))
	#		print("Zeichen mapped", charList[np.argmax(output[0])])
	#		print("Wahrscheinlichkeit", np.max(output[0]))
			str += charList[np.argmax(output[0])]

			del output
	#		print("Summe", np.sum(output[0]))
	#		print(output[0][:10])
	#		print("Länge Outputs", output.shape)
		print(str)

if __name__ == '__main__':
	main()
