# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

```
bash examples/install_requirements.sh semantic_segmentation.py

python3 examples/semantic_segmentation.py \
	--model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
	--keep_aspect_ratio \
	--output ${HOME}/segmentation_result.jpg
```
"""

import argparse
import numpy
import sys

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
import pycoral.utils.edgetpu

def main():

	#rudimentäre Ausschriften vom Coral einschalten
	pycoral.utils.edgetpu.set_verbosity(1)
	#Auflistung aller Corals
	print(pycoral.utils.edgetpu.list_edge_tpus())
	#Netz einem Coral zuweisen
	interpreter = make_interpreter("myFirstDumpNet2.tflite", device=':0')
	#Netz "initialisieren"
	interpreter.allocate_tensors()
	print("Input Details", common.input_details(interpreter, "shape"))
	data = numpy.array([[-100.0]])
	print("Input Daten", data)
	#print("Form der Input Daten", data.shape)
	#Input für das Netz setzen
	interpreter_infos = interpreter.get_input_details()[0] #['shape']
	for _ in interpreter_infos:
		print(_,  interpreter_infos[_])
	#interpreter_shape = interpreter.get_input_details()[0]['shape']
	#print("Interpreter shape", interpreter_shape)
	#data = numpy.reshape(data, interpreter_shape)
	#print("Form der Input Daten", data.shape)

	common.set_input(interpreter, data)
	print("Art des Input Tensors des Netzes", common.input_tensor(interpreter))

	interpreter.invoke()
	print("Output Tensor", common.output_tensor(interpreter, 0))

if __name__ == '__main__':
	main()
