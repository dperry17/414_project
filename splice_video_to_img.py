import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import sys

def haar_splice_map(img_gray, wavelet='haar', level=1, win=8):
    coeffs = pywt.wavedec2(img_gray, wavelet=wavelet, level=level)

    energy_maps = []

    for (LH, HL, HH) in coeffs[1:]:
        for sub in (LH, HL, HH):
            sq = sub**2
            kernel = np.ones((win, win), dtype=np.float32)
            E = cv2.filter2D(sq, -1, kernel, borderType=cv2.BORDER_REFLECT)
            E_up = cv2.resize(E, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            energy_maps.append(E_up)

    agg = np.sum(np.stack(energy_maps, axis=0), axis=0)

    #norm
    agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-12)
    return agg


if __name__ == "__main__":
    cap = cv2.VideoCapture(sys.argv[1])
    if not cap.isOpened():
        raise IOError("Cannot find file")

    ret, frame = cap.read()

    if not ret:
        raise IOError("video has no frames")

    gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0

    aggregate_map = np.zeros_like(gray_vid, dtype=np.float32)

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        score_map = haar_splice_map(gray, wavelet='haar', level=2, win=16)

        aggregate_map = np.maximum(aggregate_map, score_map)

    cap.release()

    heat_map = (aggregate_map * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_map, cv2.COLORMAP_HOT)

    cv2.imwrite("output/vid_splice.png", heat_color)
    #display
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1); plt.title('Original Gray'); plt.axis('off')
    # plt.imshow(gray, cmap='gray')
    # plt.subplot(1,3,2); plt.title('Anomaly Map'); plt.axis('off')
    # plt.imshow( splice_map, cmap='hot')
    # plt.subplot(1,3,3); plt.title('Detected Seams'); plt.axis('off')
    # plt.imshow(mask, cmap='gray')
    # plt.tight_layout()
    # plt.savefig("splices")
