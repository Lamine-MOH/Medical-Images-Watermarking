from PIL import Image
import numpy as np

# 
def text_to_binary(text):
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def text_to_image(text, path="./img/binary_img.png", image_size=(100, 100)):
    binary_string = text_to_binary(text)

    binary_len = len(binary_string)
    image_width, image_height = image_size
    total_pixels = image_width * image_height

    # Ensure we have enough pixels to represent the binary string
    if total_pixels < binary_len:
        raise ValueError("Image size too small to encode the binary string")

    # Pad the binary string if needed
    padded_binary = binary_string + '0' * (total_pixels - binary_len)

    # Reshape the binary string to fit the image size
    binary_array = np.array([int(bit) for bit in padded_binary]).reshape(image_height, image_width)

    # Convert binary array to image
    image = Image.fromarray(np.uint8(binary_array * 255))

    image.save(path)
    return image

def image_to_text(path):
    image = Image.open(path)

    binary_array = np.array(image) / 255  # Normalize pixel values to range [0, 1]
    binary_string = ''.join(str(int(pixel)) for row in binary_array for pixel in row)
    return binary_to_text(binary_string)

def binary_to_text(binary_string):
    text = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
    return text
