import numpy as np
import tensorflow as tf
import SpeechModels
from tensorflow.keras.models import Model

nCategs = 36
sr = 16000

model = SpeechModels.AttRNNSpeechModel(nCategs, samplingrate = sr, inputLength = 16000)#, rnn_func=L.LSTM)
model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
model.load_weights('models/model-attRNN.h5')

attSpeechModel = Model(inputs=model.input,
                                 outputs=[model.get_layer('output').output,
                                          model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output])


#get callable graph from model.
run_model = tf.function(lambda x: attSpeechModel(x))

# to get the concrete function from callable graph
concrete_funct = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

#convert concrete function into TF Lite model using TFLiteConverter
converter =  tf.lite.TFLiteConverter.from_concrete_functions([concrete_funct])
tflite_model = converter.convert()

# Save the model.
with open('models/cmdRecognition.tflite', 'wb') as f:
    f.write(tflite_model)