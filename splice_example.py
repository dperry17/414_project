import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import sys

def haar_splice_map(img_gray, wavelet='haar', level=1, win=8):
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=level)

    energy_maps = []
    count = 0
    for (LH, HL, HH) in coeffs[1:]:
        for sub in (LH, HL, HH):
            sq = sub**2
            kernel = np.ones((win, win), dtype=np.float32)
            E = cv2.filter2D(sq, -1, kernel, borderType=cv2.BORDER_REFLECT)
            E_up = cv2.resize(E, img_gray.shape, interpolation=cv2.INTER_LINEAR)
            energy_maps.append(E_up)
            print(len(energy_maps[count]))
            count = count + 1

    agg = np.sum(np.stack(energy_maps, axis=0), axis=0)

    #norm
    agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-12)
    return agg


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    splice_map = haar_splice_map(gray, level=2, win=16)
    _, mask = cv2.threshold(splice_map, 0.4, 1.0, cv2.THRESH_BINARY)

    #display
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title('Original Gray'); plt.axis('off')
    plt.imshow(gray, cmap='gray')
    plt.subplot(1,3,2); plt.title('Anomaly Map'); plt.axis('off')
    plt.imshow( splice_map, cmap='hot')
    plt.subplot(1,3,3); plt.title('Detected Seams'); plt.axis('off')
    plt.imshow(mask, cmap='gray')
    plt.tight_layout()
    plt.savefig("splices")
