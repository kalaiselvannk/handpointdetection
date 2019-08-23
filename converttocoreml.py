import coremltools
import keras
z=keras.models.load_model("model.h5")

coreml_model = coremltools.converters.keras.convert(model,input_names ='image',image_input_names = 'image')
coreml_model.save('my_model.mlmodel')
