import cv2
from PIL import Image
import numpy as np
import math

class Attack:
    def crop(img, cropping_value=100):
        # get the cropping dimensions
        height, width = img.shape
        left, upper, right, lower = \
            cropping_value, cropping_value, width - cropping_value, height - cropping_value

        # Crop the image
        cropped_image = img[upper:lower, left:right]

        return cropped_image

    def resize(img, width_resizing=1.5, height_resizing=1.5):
        height, width = img.shape

        img = cv2.resize(img, (int(height*height_resizing), int(width*width_resizing)))

        return img

    def shear(image, shear_factor=-0.5):
        # Get image dimensions
        height, width = image.shape

        # Define the shear matrix
        shear_matrix = np.array([
            [1, shear_factor, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        # Apply the shear transformation
        sheared_image = cv2.warpAffine(image, shear_matrix, (width, height))

        return sheared_image

    @staticmethod
    def salt(image, salt_prob=0.01):
        noisy_image = np.copy(image)
        height, width = noisy_image.shape

        # Apply salt noise
        salt_pixels = np.random.rand(height, width) < salt_prob
        noisy_image[salt_pixels] = 255

        return noisy_image
    
    @staticmethod
    def pepper(image, pepper_prob=0.01):
        noisy_image = np.copy(image)
        height, width = noisy_image.shape

        # Apply pepper noise
        pepper_pixels = np.random.rand(height, width) < pepper_prob
        noisy_image[pepper_pixels] = 0

        return noisy_image

    @staticmethod
    def gaussian_noise(image, mean=0, stddev=10):
        height, width = image.shape

        gauss = np.random.normal(mean, stddev, (height, width))
        noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def rotate90(img: np.ndarray):
        img = img.copy()
        angle = 90
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def rotate180(img: np.ndarray):
        img = img.copy()
        angle = 180
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def histogram_equalization(image):
        # Convert image to grayscale if it's not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(image)
        
        return equalized_image

    @staticmethod
    def randline(img: np.ndarray):
        img = img.copy()
        cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
        cv2.rectangle(img, (0, 0), (300, 128), (255, 0, 0), 3)
        cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
        cv2.line(img, (0, 511), (511, 0), (255, 0, 255), 5)
        return img

    @staticmethod
    def cover(img: np.ndarray):
        img = img.copy()
        cv2.circle(img, (256, 256), 63, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Just DO it ', (10, 500), font, 4, (255, 255, 0), 2)
        return img

    @staticmethod
    def compress(img, quality=10):
        # Encode the image as a JPEG buffer with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", img, encode_param)

        # Decode the JPEG buffer to get the compressed image
        compressed_image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        return compressed_image

    @staticmethod
    def quantize(image, levels=5):
        # Convert the image to grayscale if it's in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the quantization range
        min_val, max_val = np.min(image), np.max(image)
        quantization_range = max_val - min_val

        # Calculate the step size for quantization
        step_size = quantization_range / levels

        # Apply quantization to the image
        quantized_image = np.round((image - min_val) / step_size) * step_size + min_val

        return quantized_image.astype(np.uint8)

    @staticmethod
    def blur(img: np.ndarray):
        return cv2.blur(img, (10, 10))
    
    @staticmethod
    def median_filter(data, filter_size=5):
        temp = []
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))
        for i in range(len(data)):

            for j in range(len(data[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])

                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        return data_final    
    

if __name__ == "__main__":
    img = cv2.imread("./img/img_1.png", 0)
    # img = cv2.imread("./img/img_4.jpg", 0)
    # img = cv2.imread("./img/test_1.jpg", 0)

    img = Attack.crop(img)
    # img = Attack.resize(img)
    # img = Attack.resize(img, width_resizing=2, height_resizing=1)
    # img = Attack.shear(img)
    # img = Attack.salt(img)
    # img = Attack.pepper(img)
    # img = Attack.gaussian_noise(img)
    # img = Attack.rotate90(img)
    # img = Attack.rotate180(img)
    # img = Attack.histogram_equalization(img)
    # img = Attack.randline(img)
    # img = Attack.cover(img)
    # img = Attack.compress(img)
    # img = Attack.quantize(img)
    # img = Attack.blur(img)
    # img = Attack.median_filter(img)

    cv2.imwrite("./img/img_1_attacked.png", img)
