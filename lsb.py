from PIL import Image
import cv2
import numpy as np

# Convert the binary message to ASCII characters
def binary_to_text(binary_message):
    message = ''
    for i in range(0, len(binary_message), 8):
        message += chr(int(binary_message[i:i+8], 2))
        if message[-1] == '\0':  # Stop decoding when encountering null character
            break

    return message

def binary_to_gray_image_array(binary_image, image_size=200):
    image_array = []
    row = []
    col_index = 0
    row_index = 0
    for i in range(0, len(binary_image), 8):
        row.append(int(binary_image[i:i+8], 2))
        
        col_index += 1
        if col_index >= image_size:
            image_array.append(row)
            row = []

            col_index = 0
            row_index += 1

        if row_index >= image_size:
            break

    return image_array

############################################################################
def encode_lsb_rgb(cover_image, binary_watermark):
    cover_image = cover_image.convert('RGB')
    cover_pixels = cover_image.load()
    height, width = cover_image.height, cover_image.width

    index = 0
    binary_watermark_length = len(binary_watermark)
    for row in range(height):
        for col in range(width):
            r, g, b = cover_pixels[col, row]
            
            # Encode the message into the least significant bit of each color channel
            if index < binary_watermark_length:
                r = r & ~1 | int(binary_watermark[index])
                index += 1
            if index < binary_watermark_length:
                g = g & ~1 | int(binary_watermark[index])
                index += 1
            if index < binary_watermark_length:
                b = b & ~1 | int(binary_watermark[index])
                index += 1
            
            # Update the pixel with the modified color channels
            cover_pixels[col, row] = (r, g, b)

            if index >= binary_watermark_length:
                break

        if index >= binary_watermark_length:
            break

    return cover_image

def encode_lsb_text_rgb(cover_path, message, stego_path):
    # Open the image
    cover_image = Image.open(cover_path)
    
    # add the null character
    message += '\0'
    # Convert the message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # embed the watermark
    cover_image = encode_lsb_rgb(cover_image, binary_message)

    # Save the modified image with the hidden message
    cover_image.save(stego_path)

def encode_lsb_img_rgb(cover_path, watermark_path, stego_path, signature_size=200):
    # Open the cover image 
    cover_image = Image.open(cover_path)

    # Open the watermark image 
    watermark_image = Image.open(watermark_path)
    watermark_image = watermark_image.resize((signature_size, signature_size))
    watermark_image = watermark_image.convert('RGB')
    watermark_pixels = watermark_image.load()
    
    # Convert the watermark img to binary
    binary_watermark_image = ''.join(format(color, '08b') for y in range(watermark_image.height) for x in range(watermark_image.width) for color in watermark_pixels[x, y])

    # embed the watermark
    cover_image = encode_lsb_rgb(cover_image, binary_watermark_image)

    # Save the modified image with the hidden message
    cover_image.save(stego_path)


def decode_lsb_rgb(stego_image):
    stego_image = stego_image.convert('RGB')
    stego_pixels = stego_image.load()
    height, width = stego_image.height, stego_image.width

    binary_watermark = ''
    for y in range(height):
        for x in range(width):
            r, g, b = stego_pixels[x, y]
            
            # Extract the least significant bit from each color channel
            binary_watermark += str(r & 1)
            binary_watermark += str(g & 1)
            binary_watermark += str(b & 1)

    return binary_watermark

def decode_lsb_text_rgb(stego_path):
    # Open the image
    stego_image = Image.open(stego_path)
    
    # Extract the watermark
    binary_message = decode_lsb_rgb(stego_image)
    
    # Convert the binary message to ASCII characters
    return binary_to_text(binary_message)

def decode_lsb_img_rgb(stego_path, signature_path, signature_size=200):
    # Open the encoded image
    stego_image = Image.open(stego_path)

    # Extract the watermark
    binary_signature = decode_lsb_rgb(stego_image)
    binary_signature_length = len(binary_signature)
    
    # Convert the binary image to rgb format
    signature_array = []
    row = []
    col_index = 0
    row_index = 0
    for i in range(0, binary_signature_length, 24):
        if binary_signature_length-i < 24: continue

        row.append((int(binary_signature[i:i+8], 2), int(binary_signature[i+8:i+16], 2), int(binary_signature[i+16:i+24], 2)))
        
        col_index += 1
        if col_index >= signature_size:
            signature_array.append(row)
            row = []

            col_index = 0
            row_index += 1

        if row_index >= signature_size:
            break
    
    img = Image.fromarray(np.uint8(signature_array))
    img.save(signature_path)

############################################################################
def encode_lsb_gray(cover_image, binary_watermark):
    # Get the height and width of the image
    height, width = cover_image.shape
    
    # Embed the message into the cover image
    index = 0
    for row in range(height):
        for col in range(width):
            # Embed one bit of the message in the LSB of the pixel
            if index < len(binary_watermark):
                # Get the pixel value and the bit to imbed
                pixel_value = cover_image[row, col]
                bit_to_embed = int(binary_watermark[index])

                # embed the bit
                cover_image[row, col] = (pixel_value & ~1) | bit_to_embed
                index += 1
            else:
                break

        if index >= len(binary_watermark):
            break
    
    return cover_image
    
def encode_lsb_text_gray(cover_path, message, stego_path):
    # Read the image
    cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    # add the null character
    message += '\0'
    # Convert the message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # embed the watermark
    cover_image = encode_lsb_gray(cover_image, binary_message)
    
    # Save the image with the embedded message
    cv2.imwrite(stego_path, cover_image)

def encode_lsb_img_gray(cover_path, watermark_path, stego_path, signature_size=200):
    # Open cover image and secret image
    cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # resize the watermark
    watermark = cv2.resize(watermark, (signature_size, signature_size))
    
    # Convert the watermark img to binary
    binary_image = ''.join(format(pixel, '08b') for row in watermark for pixel in row)

    # embed the watermark
    cover_image = encode_lsb_gray(cover_image, binary_image)

    # Save the modified image with the hidden watermark
    cv2.imwrite(stego_path, cover_image)


def decode_lsb_gray(stego_image):
    # Get the height and width of the image
    height, width = stego_image.shape
    
    # Extract the signature
    binary_signature = ''
    for row in range(height):
        for col in range(width):
            pixel_value = stego_image[row, col]
            extracted_bit = pixel_value & 1
            binary_signature += str(extracted_bit)

    return binary_signature

def decode_lsb_text_gray(stego_path):
    # Read the image
    stego_image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract the message from the LSB of each pixel
    binary_message = decode_lsb_gray(stego_image)
    
    # Convert the binary message to ASCII characters
    return binary_to_text(binary_message)

def decode_lsb_img_gray(stego_path, signature_path, signature_size=200):
    # Open the encoded image
    stego_image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract the signature
    binary_signature = decode_lsb_gray(stego_image)
    
    # Convert the binary image to rgb format
    signature_array = binary_to_gray_image_array(binary_signature, signature_size)
    
    cv2.imwrite(signature_path, np.array(signature_array))

############################################################################
def adjust_values(a, b, c, minimum=0, maximum=255):

    boundary_a = np.array([a, max(a-1,minimum), min(a+1,maximum), max(a-2,minimum), min(a+2,maximum)])
    boundary_b = np.array([b, max(b-1,minimum), min(b+1,maximum), max(b-2,minimum), min(b+2,maximum)])
    boundary_c = np.array([c, max(c-1,minimum), min(c+1,maximum), max(c-2,minimum), min(c+2,maximum)])

    for bond_a in boundary_a:
        for bond_b in boundary_b[boundary_b != bond_a]:
            for bond_c in boundary_c[(boundary_c != bond_a) & (boundary_c != bond_b)]:
                return bond_a, bond_b, bond_c

    return "impossible","impossible","impossible"

def min_avr_max(value_1, value_2, value_3):
    if value_1 < value_2:
        if value_1 < value_3:
            min_value = value_1
            if value_2 < value_3:
                avr_value = value_2
                max_value = value_3
            else:
                avr_value = value_3
                max_value = value_2
        else:
            min_value = value_3
            avr_value = value_1
            max_value = value_2
    else:
        if value_2 < value_3:
            min_value = value_2
            if value_1 < value_3:
                avr_value = value_1
                max_value = value_3
            else:
                avr_value = value_3
                max_value = value_1
        else:
            min_value = value_3
            avr_value = value_2
            max_value = value_1

    return min_value, avr_value, max_value

def min_avr_max_reorder(min_value, avr_value, max_value, bit_1, bit_2):
    if bit_1 == 0:
        if bit_2 == 0:
            return min_value, avr_value, max_value
        else:
            return min_value, max_value, avr_value
    else:
        if bit_2 == 0:
            return max_value, min_value, avr_value
        else:
            return max_value, avr_value, min_value

def min_avr_max_bits(value_1, value_2, value_3):
    if value_1 < value_2 and value_2 < value_3: return "00"  # min avr max
    if value_3 < value_2 and value_2 < value_1: return "11"  # max avr min
    if value_1 < value_2 and value_3 < value_2: return "01"  # min max avr
    if value_3 < value_1 and value_2 < value_3: return "10"  # min avr max
        
    return ""


def encode_lsb_min_avr_max(cover_image, binary_watermark):
    # Get the height and width of the image
    height, width = cover_image.shape
    binary_watermark_length = len(binary_watermark)
    
    # Embed the message in the image using LSB
    watermark_index = 0
    for row in range(height):
        for col in range(2, width, 3):
            # Get the pixel value
            pixel_1 = cover_image[row, col-2]
            pixel_2 = cover_image[row, col-1]
            pixel_3 = cover_image[row, col]

            pixel_1, pixel_2, pixel_3 = adjust_values(pixel_1, pixel_2, pixel_3)
            
            min_value, avr_value, max_value = min_avr_max(pixel_1, pixel_2, pixel_3)

            bit_1 = int(binary_watermark[watermark_index])
            bit_2 = int(binary_watermark[watermark_index+1])

            cover_image[row, col-2], cover_image[row, col-1], cover_image[row, col] = \
                    min_avr_max_reorder(min_value, avr_value, max_value, bit_1, bit_2)

            watermark_index += 2

            if watermark_index >= binary_watermark_length:
                break

        if watermark_index >= binary_watermark_length:
            break

    return cover_image

def encode_lsb_text_min_avr_max(cover_path, message, stego_path):
    # Read the image
    cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    
    # add the null character
    message += '\0'
    # Convert the message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    
    # embed the watermark
    cover_image = encode_lsb_min_avr_max(cover_image, binary_message)
            
    # Save the image with the embedded message
    cv2.imwrite(stego_path, cover_image)

def encode_lsb_img_min_avr_max(cover_path, watermark_path, stego_path, signature_size=200):
    # Open cover image and secret image
    cover_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    watermark_image = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # resize the watermark
    watermark_image = cv2.resize(watermark_image, (signature_size, signature_size))
    
    # Convert the watermark img to binary
    binary_watermark_image = ''.join(format(pixel, '08b') for row in watermark_image for pixel in row)

    # embed the watermark
    cover_image = encode_lsb_min_avr_max(cover_image, binary_watermark_image)

    # Save the modified image with the hidden watermark
    cv2.imwrite(stego_path, cover_image)


def decode_lsb_min_avr_max(stego_image):
    # Get the height and width of the image
    height, width = stego_image.shape
    
    # Extract the signature
    binary_signature = ''
    for row in range(height):
        for col in range(2, width, 3):
            # Get the pixel value
            pixel_1 = stego_image[row, col-2]
            pixel_2 = stego_image[row, col-1]
            pixel_3 = stego_image[row, col]
            
            binary_signature += min_avr_max_bits(pixel_1, pixel_2, pixel_3)
    
    return binary_signature

def decode_lsb_text_min_avr_max(stego_path):
    # Read the image
    stego_image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract the message from the LSB of each pixel
    binary_message = decode_lsb_min_avr_max(stego_image)
    
    # Convert the binary message to ASCII characters
    return binary_to_text(binary_message)

def decode_lsb_img_min_avr_max(stego_path, signature_path, signature_size=200):
    # Open the encoded image
    stego_image = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
    
    # Extract the signature
    binary_signature = decode_lsb_min_avr_max(stego_image)
    
    # Convert the binary to image array
    signature_array = binary_to_gray_image_array(binary_signature, signature_size)
    
    cv2.imwrite(signature_path, np.array(signature_array))


# the main function
def main():
    # Example usage
    cover_path = './img/img_1.png'
    watermark_path = "./img/logo_1.jpg"
    message = 'hello this is a hidden message!!!'
    stego_path = './img/img_1_lsb.png'
    signature_path = './img/img_1_lsb_signature.png'

    #### Text Watermark ####
    # Encode the message into the image
    # encode_lsb_text_rgb(image_path, message, stego_path)
    # encode_lsb_gray(image_path, message, stego_path)
    # encode_lsb_text_min_avr_max(cover_path, message, stego_path)

    # Decode the message from the image
    # decoded_message = decode_lsb_text_rgb(stego_path)
    # decoded_message = decode_lsb_gray(stego_path)
    # decoded_message = decode_lsb_text_min_avr_max(stego_path)

    # print("Decoded message:", decoded_message)


    #### image Watermark ####
    # encode_lsb_img_gray(cover_path, watermark_path, stego_path)
    # decode_lsb_img_gray(stego_path, signature_path)
    encode_lsb_img_min_avr_max(cover_path, watermark_path, stego_path)
    decode_lsb_img_min_avr_max(stego_path, signature_path)


if __name__ == "__main__":
    main()
