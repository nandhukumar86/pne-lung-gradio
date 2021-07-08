import gradio as gr
import pydicom
import io
import tensorflow
import cv2
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.backend import log,epsilon

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1])
  union = K.sum(y_true, axis=[1]) + K.sum(y_pred, axis=[1])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred) + epsilon())

custom_objects = {"dice_coef": dice_coef}

path = 'model-dn.h5'
model = tensorflow.keras.models.load_model(path, custom_objects=custom_objects)

def detectPneumonia(file):
  f = open(file.name, mode='rb')
  raw_bytes = f.read()
  ds = pydicom.dcmread(io.BytesIO(raw_bytes), force = True)
  f.close();
  
  input_image = cv2.cvtColor(cv2.resize(ds.pixel_array,(256,256)), cv2.COLOR_BAYER_GR2RGB)
  X = []
  X.append(input_image)
  X = np.array(X)

  output_classify, output_segment =  model.predict(X)
  
  output_segment = output_segment[0].reshape(64,64)
  output_classify = "Not Identified" if output_classify[0].argmax() == 1 else "Identified"
  
  imagedata = {
      "Patient ID" : ds.PatientID,
      "Age" : ds.PatientAge,
      "Gender" : ds.PatientSex,
      "Body Part" : ds.BodyPartExamined
  }
  return imagedata, ds.pixel_array, output_segment, output_classify

iface = gr.Interface(fn = detectPneumonia, 
                     inputs = [
                              gr.inputs.File(label="DICOM File")
                     ], 
                     outputs = [
                              gr.outputs.KeyValues(label="Patient Details"),
                              gr.outputs.Image(label="Loaded Image"),
                              gr.outputs.Image(label="Segmented Image"),
                              gr.outputs.Label(label="Is Detected"),
                     ],
                     layout = "vertical",
                     title = "Pneumonia Detection",
                     allow_flagging = False)
iface.launch()
