from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import soundfile as sf
import time

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

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, INSTRUMENT_AMT, (CONV_HEIGHT, CONV_WIDTH)),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.ConvTranspose2d(INSTRUMENT_AMT, INSTRUMENT_AMT, kernel_size=(CONV_HEIGHT, CONV_WIDTH), groups=INSTRUMENT_AMT)
        self.merge = torch.nn.AvgPool3d((INSTRUMENT_AMT, 1, 1))
    def forward(self, x):
        return self.merge(self.decoder(self.encoder(x)))
    def split(self, x):
        return self.decoder(self.encoder(x))
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>8f} \n")

def main():
    p = Path(EXAMPLE_AUDIO)
    y, sr = librosa.load(p)

    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=MEL_BINS,
                                       fmax=FREQUENCY_MAX, hop_length=HOP_LENGTH)
    print(y.shape)
    print(S.shape)

    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=SAMPLE_RATE,
                                   fmax=FREQUENCY_MAX, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    #plt.show()

    x = S[None, None, :, :]
    print(x.shape)
    input_tensor = torch.Tensor(x)
    tensor_dataset = torch.utils.data.TensorDataset(input_tensor, input_tensor)
    dataloader = torch.utils.data.DataLoader(tensor_dataset)
    model = AutoEncoder().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(TRAIN_EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        start_time = time.time();
        train(dataloader, model, loss_fn, optimizer)
        test(dataloader, model, loss_fn)
        print(f"Time: {time.time() - start_time}")
    print("Done!")

    torch.save(model.state_dict(), "model.torch")

if __name__ == "__main__":
    main()