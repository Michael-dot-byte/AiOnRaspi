import argparse
import numpy as np
import sys
import time

from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import pycoral.utils.edgetpu

def main():

	charList = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	#rudimentäre Ausschriften vom Coral einschalten
	pycoral.utils.edgetpu.set_verbosity(1)

	#Auflistung aller Corals
	print(pycoral.utils.edgetpu.list_edge_tpus())

	#Netz einem Coral zuweisen
	interpreter = make_interpreter("tfKerasChars10Epochs.tflite", device=':0')
#	interpreter = make_interpreter("/home/tim/Documents/Projects/coral/pycoral/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", device=':0')

	#Netz "initialisieren"
	interpreter.allocate_tensors()

	#Input Daten für das Netz vorbereiten
	data = np.array(np.random.randint(low=0, high=255, size=(28,28))/255)

#	print("Input Daten", data)
	print("Form der Input Daten", data.shape)

	#Input für das Netz setzen
	common.set_input(interpreter, data)
	#Unterschiedliche Infos zum Input Tensor erfassen
	#print("Daten des Input Tensors des Netzes", common.input_tensor(interpreter))
	print("Input Details", common.input_details(interpreter, "shape"))
	interpreter_infos = interpreter.get_input_details()[0]
	for _ in interpreter_infos:
		print(_,  interpreter_infos[_])

	#NN laufen lassen
	startZeit= time.process_time_ns()
	interpreter.invoke()
	print("Klassifizierungszeit in µs {0}".format((time.process_time_ns() - startZeit)/1000))

	#Ergebnis vom NN abholen
	output = common.output_tensor(interpreter, 0)
	#print("Output Tensor", output)

	#Ergebnis aufbereiten
	print("Zeichen unmapped", np.argmax(output))
	print("Zeichen mapped", charList[np.argmax(output)])

if __name__ == '__main__':
	main()
