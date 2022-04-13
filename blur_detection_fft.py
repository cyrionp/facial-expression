import cv2
import imutils
import numpy as np


def detect_blur(image, threshold=15):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(img, width=500)

    size = 60

    (h, w) = img.shape
    (x, y) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    fft_shift[y - size:y + size, x - size:x + size] = 0  # Zero-out the center
    fft_shift = np.fft.ifftshift(fft_shift)  # Apply inverse shift
    recon = np.fft.ifft2(fft_shift)  # Apply inverse fft

    '''
    I have a reconstructed image, now these lines compute the magnitude 
    spectrum and mean of the magnitude values
    '''

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # print("Mean: " + str(mean))
    return True if mean < threshold else False
