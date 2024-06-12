import cv2
import numpy as np
import pywt  # PyWavelets library for wavelet transforms
from watermark import Watermark


class DWT_Watermark(Watermark):
    def __init__(self):
        self.level = 1
        self.sub_band = "HH"

    # change the level of dwt
    def set_level(self, DWT_level):
        if DWT_level > 1: self.level = DWT_level

    def set_sub_band(self, sub_band):
        self.sub_band = sub_band

    # Private method to generate embedding space for watermark embedding
    def __generate_embed_space(self, input_array):
        """
        Convert a given array into a specific format suitable for embedding.

        Args:
        - input_array (numpy.ndarray): Input array to be converted.

        Returns:
        - numpy.ndarray: Binary representation of the integer part of each element.
        - numpy.ndarray: Fractional part of each element.
        - numpy.ndarray: Indicator of whether each element is negative (1) or not (0).
        - numpy.ndarray: Signature bit extracted from the binary integer part.
        """

        # Store the shape of the input array
        array_shape = input_array.shape
        
        # Flatten the array to a 1D array
        flat_array = input_array.flatten()
        
        # Create an array indicating whether each element is negative or not
        neg_indicator = np.array([1 if flat_array[i] < 0 else 0 for i in range(len(flat_array))])

        # Take the absolute values of the elements
        abs_array = np.abs(flat_array)
        
        # Extract integer and fractional parts of each element
        int_part = np.floor(abs_array)
        frac_part = np.round(abs_array - int_part, 2)

        # Convert integer part to binary representation with 16 bits
        binary_int_part = []
        for i in range(len(int_part)):
            binary = list(bin(int(int_part[i]))[2:])  # Convert integer to binary string
            padded_binary = [0] * (16 - len(binary))           # Pad with zeros to make it 16 bits long
            padded_binary.extend(binary)
            binary_int_part.append(np.array(padded_binary, dtype=np.uint16))
        binary_int_part = np.array(binary_int_part)

        # Extract the 11th bit (index 10) from each binary integer part as the signature bit
        signature_bit = []
        for i in range(len(binary_int_part)):
            signature_bit.append(binary_int_part[i][10])
        signature_bit = np.array(signature_bit).reshape(array_shape)

        # Reshape arrays to match the shape of the original array and return
        return np.array(binary_int_part), frac_part.reshape(array_shape), neg_indicator.reshape(array_shape), signature_bit

    # Private method to embed watermark signature into embedding space
    def __embed_signature(self, binary_integer_part, fractional_part, negative_indicator, signature):
        """
        Embed a signature into the binary integer part of a number, combine it with the fractional part, and adjust the sign if necessary.

        Args:
        - binary_integer_part (numpy.ndarray): Binary representation of the integer part of each number.
        - fractional_part (numpy.ndarray): Fractional part of each number.
        - negative_indicator (numpy.ndarray): Indicator of whether each number is negative.
        - signature (numpy.ndarray): Signature bit to embed into the binary integer part.

        Returns:
        - numpy.ndarray: Combined and adjusted array.
        """

        # Store the shape of the fractional part array for later reshaping
        shape = fractional_part.shape

        # Flatten arrays for easier manipulation
        fractional_part_flat = fractional_part.flatten()
        negative_indicator_flat = negative_indicator.flatten()

        signature_length = len(signature)  # Length of the signature
        binary_integer_part_length = len(binary_integer_part)  # Number of elements in binary_integer_part

        # Embed signature into binary integer part
        if signature_length >= binary_integer_part_length:
            for i in range(binary_integer_part_length):
                binary_integer_part[i][10] = signature[i]
        if signature_length < binary_integer_part_length:
            repetition_rate = binary_integer_part_length // signature_length  # Repetition rate
            for i in range(signature_length):
                for j in range(repetition_rate):
                    binary_integer_part[i + j * signature_length][10] = signature[i]

        # Convert binary integer part back to float and combine with fractional part
        combined_integer_part = []
        for i in range(len(binary_integer_part)):
            binary_string = '0b' + ''.join([str(bit) for bit in binary_integer_part[i]])  # Convert binary to string
            combined_integer_part.append(eval(binary_string))  # Evaluate string as binary to decimal

        combined_array = np.array(combined_integer_part) + np.array(fractional_part_flat)

        # Adjust sign if necessary
        adjusted_array = np.array([-1 * combined_array[i] if negative_indicator_flat[i] == 1 else combined_array[i]
                                for i in range(len(combined_array))]).reshape(shape)

        return adjusted_array.reshape(shape)


    # Private method to extract signature from embedding space
    def __extract_signature(self, embedding_space_signature, signature_length):
        """
        Private method to extract signature from the embedding space.

        Args:
        - embedding_space_signature (numpy.ndarray): Embedding space signature to extract.
        - signature_length (int): Length of the signature to be extracted.

        Returns:
        - list: List of extracted signatures.
        """

        # Flatten and convert to list for easier manipulation
        flattened_signature = list(embedding_space_signature.flatten())

        m = len(flattened_signature)  # Length of the embedding space signature
        n = signature_length  # Length of the signature to be extracted
        extracted_signatures = []

        # Case where the desired signature length is greater than the available signature length
        if n > m:
            extracted_signatures.append(flattened_signature + ([0] * (n - m)))

        # Case where the desired signature length is less than or equal to the available signature length
        if n <= m:
            repetition_rate = m // n  # Repetition rate
            for i in range(repetition_rate):
                extracted_signatures.append(flattened_signature[i * n: (i + 1) * n])

        return extracted_signatures

    # Method to embed watermark into image
    def embed_watermark(self, cover_image, watermark_signature):
        """
        Method to embed a watermark into an image using Discrete Wavelet Transform (DWT).

        Args:
        - image (numpy.ndarray): Input image to embed the watermark into.
        - watermark_signature (numpy.ndarray): Signature of the watermark to embed.

        Returns:
        - numpy.ndarray: Image with embedded watermark.
        """

        # Get the dimensions of the image
        width, height = cover_image.shape[:2]

        # Perform Discrete Wavelet Transform (DWT) with Haar wavelet
        wavelets = []

        # Perform DWT on the original image
        LL, (HL, LH, HH) = pywt.dwt2(np.array(cover_image[:32 * (width // 32), :32 * (height // 32)]), 'haar')
        wavelets.append({"LL": LL, "HL": HL, "LH": LH, "HH": HH})

        # Perform further DWT levels
        for i in range(self.level):
            LL, (HL, LH, HH) = pywt.dwt2(wavelets[i]["LL"], 'haar')
            wavelets.append({"LL": LL, "HL": HL, "LH": LH, "HH": HH})

        # Generate embedding space for watermark
        binary_integer_part, fractional_part, combo_negative_indicator, _ = self.__generate_embed_space(wavelets[-1][self.sub_band])
        wavelets[-1][self.sub_band] = self.__embed_signature(binary_integer_part, fractional_part, combo_negative_indicator, watermark_signature)

        # Perform inverse DWT to reconstruct the image
        for i in range(self.level-1, -1, -1):
            wavelets[i]["LL"] = pywt.idwt2((wavelets[i+1]["LL"], (wavelets[i+1]["HL"], wavelets[i+1]["LH"], wavelets[i+1][self.sub_band])), 'haar')

        # Reconstruct the image from the wavelet coefficients
        cover_image[:32 * (width // 32), :32 * (height // 32)] = pywt.idwt2((wavelets[0]["LL"], (wavelets[0]["HL"], wavelets[0]["LH"], wavelets[0][self.sub_band])), 'haar')

        return cover_image


    # Method to extract watermark signature from image
    def extract_watermark_signature(self, image):
        """
        Method to extract watermark signature from an image using Discrete Wavelet Transform (DWT).

        Args:
        - image (numpy.ndarray): Input image to extract the watermark signature from.

        Returns:
        - numpy.ndarray: Extracted watermark signature.
        """

        # Get the dimensions of the image
        width, height = image.shape[:2]

        # Perform Discrete Wavelet Transform (DWT) with Haar wavelet
        wavelets = []

        # Perform DWT on the original image
        LL, (HL, LH, HH) = pywt.dwt2(image[:32 * (width // 32), :32 * (height // 32)], 'haar')
        wavelets.append({"LL": LL, "HL": HL, "LH": LH, "HH": HH})

        # Perform further DWT levels
        for i in range(self.level):
            LL, (HL, LH, HH) = pywt.dwt2(wavelets[i]["LL"], 'haar')
            wavelets.append({"LL": LL, "HL": HL, "LH": LH, "HH": HH})

        # Generate embedding space for watermark signature
        _, _, _, original_signature = self.__generate_embed_space(wavelets[-1][self.sub_band])

        # Extract watermark signature from embedding space
        extracted_signature = self.__extract_signature(original_signature, self.signature_size**2)
        
        return extracted_signature



if __name__ == "__main__":

    # Define file paths for input and output images
    cover_path = "./img/img_4.jpg"
    watermark_path = "./img/logo_1.jpg"
    output_path = "./img/img_4_dwt.jpg"
    output_attacked_path = "./img/img_4_dwt_attached.jpg"
    signature_path = "./img/img_4_dwt_signature.jpg"

    # Read the cover image and watermark image
    img = cv2.imread(cover_path)
    wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    # Create an instance of DWT_Watermark
    dwt = DWT_Watermark()

    # Embed the watermark into the cover image
    wmd = dwt.embed(img, wm)
    cv2.imwrite(output_path, wmd)

    # Read the watermarked image
    img = cv2.imread(output_path)

    # Perform various attacks on the watermarked image (commented out)
    # img = Attack.gray(img)
    # img = Attack.rotate180(img)
    # img = Attack.chop5(img)
    # img = Attack.saltnoise(img)
    # img = Attack.randline(img)
    # img = Attack.cover(img)
    # img = Attack.brighter10(img)
    # img = Attack.darker10(img)
    # img = Attack.largersize(img)
    # img = Attack.smallersize(img)
    # cv2.imwrite(output_attacked_path, img)


    # Create a new instance of DWT_Watermark
    dwt = DWT_Watermark()

    # Extract the watermark signature from the attacked image
    signature = dwt.extract(img)

    # Save the extracted signature as an image
    cv2.imwrite(signature_path, signature)
