import tensorflow as tf
import pyaudio
import wave

sample_format = pyaudio.paInt16
channels = 1
sr = 16000
chunk = 1000
filename = "output.wav"

def record_(seconds):
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=sr,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(sr / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sr)
    wf.writeframes(b''.join(frames))
    wf.close()
    return 0


def load_tflite_model(file):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    return interpreter

def predict(model, sample):
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    input_data = sample
    model.set_tensor(input_details[0]['index'], input_data)

    model.invoke()

    # The function `get_tensor()` returns a copy of the tensor data. # Use `tensor()` in order to get a pointer to the tensor.
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data
