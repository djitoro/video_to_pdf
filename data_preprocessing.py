import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os


# oversampling: sample rate changes
# base value is 16000
def resample_audio(audio_path, target_sr=16000):
	# sr - Sample rate; y - new sound
	y, sr = librosa.load(audio_path, sr=target_sr)
	return y, sr


# removing background noise
def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
	nyquist = 0.5 * sample_rate
	normal_cutoff = cutoff_freq / nyquist
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	filtered_data = filtfilt(b, a, data)
	print(f"Filtered audio shape: {filtered_data.shape}")
	return filtered_data


# Convert audio data to the model’s expected input
def convert_to_model_input(y, target_length):
	if len(y) < target_length:
		y = np.pad(y, (0, target_length - len(y)))
	else:
		y = y[:target_length]
	return y


'''
sample_audio_path = 'barbie_4.wav'

resampled_audio, sr = resample_audio(sample_audio_path)
print(f"Sample rate after Resampling: {sr}")

filtered_audio = butter_lowpass_filter(resampled_audio, cutoff_freq=4000, sample_rate=sr)

model_input = convert_to_model_input(filtered_audio, target_length=16000)
print(f"Model input shape: {model_input.shape}")
'''


def stream_audio_dataset(dataset_path, batch_size=32, target_length=16000, target_sr=None):
	#  Get all audio file paths in the dataset path
	audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_path) for file in files]

	# Shuffle the audio files for randomness
	np.random.shuffle(audio_files)

	# Yield batches of audio data
	for i in range(0, len(audio_files), batch_size):
		batch_paths = audio_files[i:i + batch_size]
		batch_data = []

		for file_path in batch_paths:
			# Load and preprocess each audio file
			y, sr = librosa.load(file_path, sr=target_sr)

			# Resampling
			if target_sr is not None and sr != target_sr:
				y = librosa.resample(y, sr, target_sr)
				sr = target_sr

			filtered_audio = butter_lowpass_filter(y, cutoff_freq=4000, sample_rate=sr)
			model_input = convert_to_model_input(filtered_audio, target_length=target_length)
			batch_data.append(model_input)

		yield np.array(batch_data)


# Load the dataset folder
dataset_path = 'barbie'

for batch_data in stream_audio_dataset(dataset_path, batch_size=2, target_sr=16000):
	# Process each batch of audio data
	print(f"Processing batch with {len(batch_data)} files")

	# Print the shape of the first file in the batch
	print(f"Shape of the first file: {batch_data[0].shape}")

