from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf

EXAMPLE_AUDIO = Path('akasha2.wav')
INSTRUMENT_AMT = 10
CONV_HEIGHT = 96
CONV_WIDTH = 8
MEL_BINS = 128

p = Path(EXAMPLE_AUDIO)
y, sr = librosa.load(p, duration=30)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=MEL_BINS,
                                   fmax=8192)
print(y.shape)
print(S.shape)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis='mel', sr=sr,
                               fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

x = S[None, :, :, None]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(INSTRUMENT_AMT, (CONV_HEIGHT, CONV_WIDTH),  activation='relu'),
    tf.keras.layers.Conv2DTranspose(INSTRUMENT_AMT, (CONV_HEIGHT, CONV_WIDTH), activation='relu'),
    tf.keras.layers.Reshape((MEL_BINS, 1292, INSTRUMENT_AMT, 1)),
    tf.keras.layers.AveragePooling3D(pool_size=(1, 1, INSTRUMENT_AMT)),
    tf.keras.layers.Reshape((MEL_BINS, 1292, 1))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy']
)

model(x)
model.summary()

model.fit(x, x, batch_size=1, epochs=1000)

out = model(x)
out = np.squeeze(out)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(out, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis='mel', sr=sr,
                               fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

out = librosa.feature.inverse.mel_to_audio(out)
sf.write('out.wav', out, sr, subtype='PCM_24')

intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('conv2d_transpose').output)
out = intermediate_layer_model(x)
out = np.squeeze(out)
print(out.shape)
for i in range(INSTRUMENT_AMT):
    out_i = librosa.feature.inverse.mel_to_audio(out[:, :, i])
    sf.write(f'out{i}.wav', out_i, sr, subtype='PCM_24')