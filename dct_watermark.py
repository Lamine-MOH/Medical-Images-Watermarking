import cv2
import numpy as np
from watermark import Watermark


# Class for embedding and extracting watermarks using DCT
class DCT_Watermark(Watermark):
    def __init__(self):
        # Set quantization factor and block size
        self.Q = 10
        self.block_size = 2

    def set_block_size(self, block_size):
        self.block_size = block_size

    # Method to embed watermark into image using DCT
    def embed_watermark(self, cover_image: np.ndarray, watermark_image):
        # Get dimensions of the image
        wight, height = cover_image.shape[:2]

        # Define positions for embedding watermark
        embed_pos = [(0, 0)]
        if wight > 2 * self.signature_size * self.block_size:
            embed_pos.append((wight-self.signature_size*self.block_size, 0))
        if height > 2 * self.signature_size * self.block_size:
            embed_pos.append((0, height-self.signature_size*self.block_size))
        if len(embed_pos) == 3:
            embed_pos.append((wight-self.signature_size*self.block_size, height-self.signature_size*self.block_size))

        # Iterate through embedding positions
        for x, y in embed_pos:
            for i in range(x, x+self.signature_size * self.block_size, self.block_size):
                for j in range(y, y+self.signature_size*self.block_size, self.block_size):
                    # Get DCT of the block
                    block = np.float32(cover_image[i:i + self.block_size, j:j + self.block_size])
                    if block.shape[0] < self.block_size or block.shape[1] < self.block_size: continue
                    block = cv2.dct(block)
                    
                    # Modify DCT coefficient according to the watermark
                    block[self.block_size-1, self.block_size-1] = self.Q * \
                        watermark_image[((i-x)//self.block_size) * self.signature_size + (j-y)//self.block_size]
                    
                    # Perform inverse DCT
                    block = cv2.idct(block)

                    # Adjust pixel values to ensure validity
                    maximum = max(block.flatten())
                    minimum = min(block.flatten())
                    if maximum > 255:
                        block = block - (maximum - 255)
                    if minimum < 0:
                        block = block - minimum

                    # Update the image with the modified block
                    cover_image[i:i+self.block_size, j:j+self.block_size] = block
        
        return cover_image

    # Method to extract watermark from image using DCT
    def extract_watermark_signature(self, stego_image):
        # Initialize extracted signature
        extracted_signature = np.zeros(self.signature_size**2)

        # Iterate through image blocks
        for i in range(0, self.signature_size * self.block_size, self.block_size):
            for j in range(0, self.signature_size * self.block_size, self.block_size):
                # Get DCT of the block
                block = np.float32(stego_image[i:i+self.block_size, j:j+self.block_size])
                if block.shape[0] < self.block_size or block.shape[1] < self.block_size: continue
                block = cv2.dct(block)

                # Check the DCT coefficient corresponding to the bottom-right corner of the block
                if block[self.block_size-1, self.block_size-1] > self.Q / 2:
                    extracted_signature[(i//self.block_size) * self.signature_size + j//self.block_size] = 1
        
        return [extracted_signature]
    

if __name__ == "__main__":
    message = """
Patient ID: 123456
Name: John Doe
Date of Birth: January 15, 1980
Gender: Male
Address: 123 Elm Street, Springfield, IL, 62701
Phone Number: (555) 123-4567
Email: johndoe@example.com
Emergency Contact: Jane Doe - (555) 765-4321
Primary Care Physician: Dr. Emily Smith
Insurance Provider: HealthFirst Insurance
Policy Number: HF123456789
Medical History: Hypertension, Type 2 Diabetes, Asthma
Current Medications: Metformin, Lisinopril, Albuterol Inhaler
Allergies: Penicillin, Peanuts
Recent Procedures: Appendectomy (March 2023), Colonoscopy (January 2024)
Upcoming Appointments: Cardiology Check-up on June 10, 2024
Notes: Patient experiences occasional dizziness, advised to monitor blood pressure regularly."""

    cover_path = "./img/img_4.jpg"
    watermark_path = "./img/logo_1.jpg"
    output_path = "./img/img_4_dct.jpg"
    output_attacked_path = "./img/img_4_dct_attached.jpg"
    signature_path = "./img/img_4_dct_signature.jpg"

    cover_image = cv2.imread(cover_path)

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

    dct = DCT_Watermark()
    stego_image = dct.embed(cover_image, watermark)
    cv2.imwrite(output_path, stego_image)

    cover_image = cv2.imread(output_path)

    dct = DCT_Watermark()
    signature = dct.extract(cover_image)
    cv2.imwrite(signature_path, signature)
    