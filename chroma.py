import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import stft

def freq_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)

def extract_chromagram(file_path, plot=False):
    # Load audio
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # STFT parameters
    window_size = 10000
    hop_length = 10000
    window = 'blackman'
    noverlap = window_size - hop_length

    # STFT
    f, t, Zxx = stft(y, fs=sr, nperseg=window_size, noverlap=noverlap, window=window)
    magnitude = np.abs(Zxx)

    # Filter out DC bin
    valid = f > 0
    valid_freqs = f[valid]
    valid_magnitude = magnitude[valid, :]
    midi = freq_to_midi(valid_freqs)
    midi_rounded = np.round(midi).astype(int)

    # Convert to pitch energies
    min_midi = 33
    max_midi = 84
    midi_range = max_midi - min_midi + 1
    pitch_energies = np.zeros((midi_range, valid_magnitude.shape[1]))

    for bin_idx, midi_note in enumerate(midi_rounded):
        if min_midi <= midi_note <= max_midi:
            pitch_energies[midi_note - min_midi, :] += valid_magnitude[bin_idx, :]

    # Fold to chromagram
    chromagram = np.zeros((12, valid_magnitude.shape[1]))
    for i in range(midi_range):
        chroma_bin = (i + min_midi) % 12
        chromagram[chroma_bin, :] += pitch_energies[i, :]

    # Normalize
    chromagram /= np.max(chromagram) + 1e-6

    # Optional plotting
    if plot:
        time_axis = np.linspace(0, len(y) / sr, num=len(y))

        fig, axs = plt.subplots(4, 1, figsize=(12, 18))
        axs[0].plot(time_axis, y)
        axs[0].set_title('a. Signal Amplitude vs Time')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')

        img1 = axs[1].imshow(20 * np.log10(magnitude + 1e-6), origin='lower', aspect='auto',
                             extent=[t[0], t[-1], f[0], f[-1]], cmap='magma')
        axs[1].set_title('b1. STFT Spectrogram (Linear Frequency)')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_xlabel('Time (s)')
        fig.colorbar(img1, ax=axs[1])


        img3 = axs[2].imshow(pitch_energies, origin='lower', aspect='auto',
                             extent=[t[0], t[-1], min_midi, max_midi], cmap='viridis')
        axs[2].set_title('c. Pitch Energies (per MIDI note)')
        axs[2].set_ylabel('MIDI Note')
        axs[2].set_xlabel('Time (s)')
        fig.colorbar(img3, ax=axs[2])

        img4 = axs[3].imshow(chromagram, origin='lower', aspect='auto',
                             extent=[t[0], t[-1], 0, 12])
        axs[3].set_yticks(np.arange(12))
        axs[3].set_yticklabels(['C', 'C#', 'D', 'D#', 'E', 'F',
                                'F#', 'G', 'G#', 'A', 'A#', 'B'])
        axs[3].set_title('d. Chromagram')
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Pitch Class')
        fig.colorbar(img4, ax=axs[3])

        plt.tight_layout()
        plt.show()

    return chromagram, t




