import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from process import AutoEncoder

EXAMPLE_AUDIO = Path('input/akasha3.wav')
INSTRUMENT_AMT = 5
CONV_HEIGHT = 1024
CONV_WIDTH = 4
MEL_BINS = 2048
SAMPLE_RATE = 44100
FREQUENCY_MAX = 16384
TRAIN_EPOCHS = 100
HOP_LENGTH = 256


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def main():
    p = Path(EXAMPLE_AUDIO)
    y, _ = librosa.load(p)

    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=MEL_BINS,
                                       fmax=FREQUENCY_MAX, hop_length=HOP_LENGTH)
    
    S = S[None, None, :, :]
    dataTensor = torch.Tensor(S)

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load("model.torch", map_location=device))
    model.eval()
    output_tensor = model.forward(dataTensor)
    
    out = output_tensor.detach().numpy();
    out = out.squeeze();

    out = librosa.feature.inverse.mel_to_audio(out, fmax=FREQUENCY_MAX, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    sf.write('output/out_torch.wav', out, 22050, subtype='PCM_24')
    
    output_tensor = model.split(dataTensor)
    out = output_tensor.detach().numpy();
    print(out.shape)
    
    for i in range(INSTRUMENT_AMT):
        out_i = librosa.feature.inverse.mel_to_audio(out[0, i, :, :], fmax=FREQUENCY_MAX, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        sf.write(f'output/out_torch_{i}.wav', out_i, 22050, subtype='PCM_24')

if __name__ == "__main__":
    main()