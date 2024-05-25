import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def snr(image, noisy_image):
    signal_power = np.mean(image ** 2)
    noise_power = max(np.mean((image - noisy_image) ** 2), 0.0000001)
    return 10 * np.log10(signal_power / noise_power)

def psnr(image, noisy_image):
    max_pixel = 255.0
    mse_value = max(mse(image, noisy_image), 0.0000001)
    return 10 * np.log10((max_pixel ** 2) / mse_value)

def nc(watermark, signature, type="image"):
    if type == "image":
        cross_correlation = np.sum((watermark - np.mean(watermark)) * (signature - np.mean(signature)))
        nc = cross_correlation / (np.std(watermark) * np.std(signature) * watermark.size)
    else:
        # Convert messages to binary
        original_binary = ''.join(format(ord(c), '08b') for c in watermark)
        extracted_binary = ''.join(format(ord(c), '08b') for c in signature)

        cross_correlation = sum(int(m) * int(e) for m, e in zip(original_binary, extracted_binary))

        # Calculate normalization terms
        norm_m = sum(int(m)**2 for m in original_binary) ** 0.5
        norm_e = sum(int(e)**2 for e in extracted_binary) ** 0.5

        nc = cross_correlation / max((norm_m * norm_e), 0.0000001)
        
    return nc

def calculate_ssim(image1, image2):
    return ssim(image1, image2, multichannel=True)


def main():
    # Example usage:
    # Read two images
    cover = cv2.imread("img/img_1.png", cv2.IMREAD_GRAYSCALE)
    stego = cv2.imread("img/img_1_DCT.png", cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread("img/logo_1.jpg", cv2.IMREAD_GRAYSCALE)
    signature = cv2.imread("img/img_1_DCT_signature.png", cv2.IMREAD_GRAYSCALE)

    # Calculate metrics
    print("MSE:", mse(cover, stego))
    print("SNR:", snr(cover, stego))
    print("PSNR:", psnr(cover, stego))
    print("NC:", nc(watermark, signature))
    print("SSIM:", calculate_ssim(cover, stego))


if __name__ == "__main__":
    main()
