import gradio as gr
import pydicom
import io

def detectPneumonia(file):
  f = open(file.name, mode='rb')
  raw_bytes = f.read()
  ds = pydicom.dcmread(io.BytesIO(raw_bytes), force = True)
  f.close();
  imagedata = {
      "Patient ID" : ds.PatientID,
      "Age" : ds.PatientAge,
      "Gender" : ds.PatientSex,
      "Body Part" : ds.BodyPartExamined
  }
  return imagedata, ds.pixel_array

iface = gr.Interface(fn = detectPneumonia, 
                     inputs = [
                              gr.inputs.File(label="DICOM File")
                     ], 
                     outputs = [
                              gr.outputs.KeyValues(label="Patient Details"),
                              gr.outputs.Image(label="Loaded Image")
                     ],
                     layout = "vertical",
                     title = "Pneumonia Detection",
                     allow_flagging = False)
iface.launch()
