from PIL import Image, ImageTk
from PIL import ImageTk    
import os
import lsb as lsb
from metrics import *
from text_to_img import *
from dct_watermark import *
from dwt_watermark import *
from attack import *
import json

def apply_attacks(img):
    attacked_images = {
        # "crop": Attack.crop(img), 
        # "resize": Attack.resize(img), 
        # "resize": Attack.resize(img, width_resizing=2, height_resizing=1), 
        # "shear": Attack.shear(img), 
        "salt": Attack.salt(img), 
        "pepper": Attack.pepper(img), 
        "gaussian_noise": Attack.gaussian_noise(img), 
        # "rotate90": Attack.rotate90(img), 
        # "rotate180": Attack.rotate180(img), 
        "histogram_equalization": Attack.histogram_equalization(img), 
        "randline": Attack.randline(img), 
        "cover": Attack.cover(img), 
        "compress": Attack.compress(img), 
        "quantize": Attack.quantize(img), 
        # "blur": Attack.blur(img), 
        # "median_filter": Attack.median_filter(img)
    }

    return attacked_images


def lsb_text_rgb(dataset_folders, message, export_path="./dataset_export/", results_path="./dataset_results/", limit=200):
    print(f"\n\nAlgorithm LSB Text RGB")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_RGB.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_text_rgb(cover_path, message, stego_path)
            decoded_message = lsb.decode_lsb_text_rgb(stego_path)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(message, decoded_message, type="text"),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_LSB_RGB_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_extracted_message = lsb.decode_lsb_text_rgb(attacked_stego_path)

                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(message, attacked_extracted_message, type="text"), 7),
                }
            
        print("\n")

    with open(f"{results_path}LSB_Text_RGB.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

def lsb_img_rgb(dataset_folders, watermark_path, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=200):
    print(f"\n\nAlgorithm LSB Text RGB")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_Image_RGB.png"
            signature_path = f"{export_path}{folder}-{i}_LSB_RGB_signature.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_img_rgb(cover_path, watermark_path, stego_path)
            lsb.decode_lsb_img_rgb(stego_path, signature_path, signature_size=signature_size)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            metrics[stego_path] = {
                "MSE": round(mse(cover_img, stego_img), 7),
                "SNR": round(snr(cover_img, stego_img), 7),
                "PSNR": round(psnr(cover_img, stego_img), 7),
                "SSIM": round(calculate_ssim(cover_img, stego_img), 7),
                "NC": round(nc(watermark, signature), 7),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_LSB_Image_RGB_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_signature_path = f"{export_path}{folder}-{i}_LSB_RGB_{attack_name}_signature.png"  
                lsb.decode_lsb_img_rgb(attacked_stego_path, attacked_signature_path)

                # Calculate
                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))
                
                attacked_signature = cv2.imread(attacked_signature_path, 0)
                attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(watermark, attacked_signature), 7),
                }
            
        print("\n")

    with open(f"{results_path}LSB_Image_RGB.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)


def lsb_text_gray(dataset_folders, message, export_path="./dataset_export/", results_path="./dataset_results/", limit=200):
    print(f"\n\nAlgorithm LSB Text Gray")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_Gray.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_text_gray(cover_path, message, stego_path)
            decoded_message = lsb.decode_lsb_text_gray(stego_path)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(message, decoded_message, type="text"),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_LSB_Gray_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_extracted_message = lsb.decode_lsb_text_gray(attacked_stego_path)

                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(message, attacked_extracted_message, type="text"), 7),
                }
            
        print("\n")

    with open(f"{results_path}LSB_Text_Gray.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

def lsb_img_gray(dataset_folders, watermark_path, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=200):
    print(f"\n\nAlgorithm LSB Text Gray")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_Image_Gray.png"
            signature_path = f"{export_path}{folder}-{i}_LSB_Gray_signature.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_img_gray(cover_path, watermark_path, stego_path)
            lsb.decode_lsb_img_gray(stego_path, signature_path, signature_size=signature_size)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            metrics[stego_path] = {
                "MSE": round(mse(cover_img, stego_img), 7),
                "SNR": round(snr(cover_img, stego_img), 7),
                "PSNR": round(psnr(cover_img, stego_img), 7),
                "SSIM": round(calculate_ssim(cover_img, stego_img), 7),
                "NC": round(nc(watermark, signature), 7),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_LSB_Image_Gray_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_signature_path = f"{export_path}{folder}-{i}_LSB_Gray_{attack_name}_signature.png"  
                lsb.decode_lsb_img_gray(attacked_stego_path, attacked_signature_path)

                # Calculate
                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))
                
                attacked_signature = cv2.imread(attacked_signature_path, 0)
                attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(watermark, attacked_signature), 7),
                }
            
        print("\n")

    with open(f"{results_path}LSB_Image_Gray.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

 
def lsb_text_min_avr_max(dataset_folders, message, export_path="./dataset_export/", results_path="./dataset_results/", limit=200):
    print(f"\n\nAlgorithm LSB Text Min Avr Max")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_Min_Avr_Max.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_text_min_avr_max(cover_path, message, stego_path)
            decoded_message = lsb.decode_lsb_text_min_avr_max(stego_path)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(message, decoded_message, type="text"),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_LSB_Min_Avr_Max_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_extracted_message = lsb.decode_lsb_text_min_avr_max(attacked_stego_path)

                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(message, attacked_extracted_message, type="text"), 7),
                }
            
        print("\n")

    with open(f"{results_path}LSB_Text_Min_Avr_Max.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

def lsb_img_min_avr_max(dataset_folders, watermark_path, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=200):
    print(f"\n\nAlgorithm LSB Text Min Avr Max")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_LSB_Image_Min_Avr_Max.png"
            signature_path = f"{export_path}{folder}-{i}_LSB_Min_Avr_Max_signature.png"

            # Encode and Decode the text watermark
            lsb.encode_lsb_img_min_avr_max(cover_path, watermark_path, stego_path)
            lsb.decode_lsb_img_min_avr_max(stego_path, signature_path, signature_size=signature_size)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            metrics[stego_path] = {
                "MSE": round(mse(cover_img, stego_img), 7),
                "SNR": round(snr(cover_img, stego_img), 7),
                "PSNR": round(psnr(cover_img, stego_img), 7),
                "SSIM": round(calculate_ssim(cover_img, stego_img), 7),
                "NC": round(nc(watermark, signature), 7),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                try:
                    attacked_stego_path = f"{export_path}{folder}-{i}_LSB_Image_Min_Avr_Max_{attack_name}.png"  
                    cv2.imwrite(attacked_stego_path, attacked_stego_img)

                    # decode watermark from attacked stego image
                    attacked_signature_path = f"{export_path}{folder}-{i}_LSB_Min_Avr_Max_{attack_name}_signature.png"  
                    lsb.decode_lsb_img_min_avr_max(attacked_stego_path, attacked_signature_path, signature_size)

                    # Calculate
                    attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                    attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))
                    
                    attacked_signature = cv2.imread(attacked_signature_path, 0)
                    attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

                    attacked_metrics[stego_path][attack_name] = {
                        "MSE": round(mse(cover_img, attacked_stego_img), 7),
                        "SNR": round(snr(cover_img, attacked_stego_img), 7),
                        "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                        "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                        "NC": round(nc(watermark, attacked_signature), 7),
                    }
                except:
                    pass
            
        print("\n")

    with open(f"{results_path}LSB_Image_Min_Avr_Max.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)


def dct_text(dataset_folders, message, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=80, block_size=2):
    print(f"\n\nAlgorithm DCT Text")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_DCT.png"
            signature_path = f"{export_path}{folder}-{i}_DCT_signature.png"

            # Encode and Decode the text watermark
            watermark_path = f"{export_path}binary_img.png"
            text_to_image(message, path=watermark_path, image_size=(signature_size, signature_size))

            cover_img = cv2.imread(cover_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

            dct = DCT_Watermark()
            dct.set_signature_size(signature_size)
            dct.set_block_size(block_size)
            
            stego_img = dct.embed(cover_img, watermark_img)
            cv2.imwrite(stego_path, stego_img)

            signature = dct.extract(stego_img)
            cv2.imwrite(signature_path, signature)

            decoded_message = image_to_text(signature_path)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(message, decoded_message, type="text"),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_DCT_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_stego_image = cv2.imread(attacked_stego_path)
                attacked_signature = dct.extract(attacked_stego_image)

                attacked_signature_path = f"{export_path}{folder}-{i}_DCT_{attack_name}_signature.png"  
                cv2.imwrite(attacked_signature_path, attacked_signature)

                attacked_extracted_message = image_to_text(attacked_signature_path)

                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(message, attacked_extracted_message, type="text"), 7),
                }
            
        print("\n")

    with open(f"{results_path}DCT_Text.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

def dct_img(dataset_folders, watermark_path, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=200, block_size=2):
    print(f"\n\nAlgorithm DCT Image")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_DCT_Image.png"
            signature_path = f"{export_path}{folder}-{i}_DCT_Image_signature.png"

            # Encode and Decode the text watermark
            cover_img = cv2.imread(cover_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

            dct = DCT_Watermark()
            dct.set_signature_size(signature_size)
            dct.set_block_size(block_size)
            
            stego_img = dct.embed(cover_img, watermark_img)
            cv2.imwrite(stego_path, stego_img)

            signature = dct.extract(stego_img)
            cv2.imwrite(signature_path, signature)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(watermark_img, signature),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_DCT_Image_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_stego_image = cv2.imread(attacked_stego_path)
                attacked_signature = dct.extract(attacked_stego_image)

                attacked_signature_path = f"{export_path}{folder}-{i}_DCT_Image_{attack_name}_signature.png"  
                cv2.imwrite(attacked_signature_path, attacked_signature)

                # Calculate
                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))
                
                attacked_signature = cv2.imread(attacked_signature_path, 0)
                attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(watermark_img, attacked_signature), 7),
                }
            
        print("\n")

    with open(f"{results_path}DCT_Image.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)



def dwt_text(dataset_folders, message, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=80, wavelet_level=1):
    print(f"\n\nAlgorithm DWT Text")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_DWT_Text.png"
            signature_path = f"{export_path}{folder}-{i}_DWT_Text_signature.png"

            # Encode and Decode the text watermark
            watermark_path = f"{export_path}binary_img.png"
            text_to_image(message, path=watermark_path, image_size=(signature_size, signature_size))

            cover_img = cv2.imread(cover_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

            dwt = DWT_Watermark()
            dwt.set_signature_size(signature_size)
            dwt.set_level(wavelet_level)
            
            stego_img = dwt.embed(cover_img, watermark_img)
            cv2.imwrite(stego_path, stego_img)

            signature = dwt.extract(stego_img)
            cv2.imwrite(signature_path, signature)

            decoded_message = image_to_text(signature_path)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(message, decoded_message, type="text"),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_DWT_Text_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_stego_image = cv2.imread(attacked_stego_path)
                attacked_signature = dwt.extract(attacked_stego_image)

                attacked_signature_path = f"{export_path}{folder}-{i}_DWT_Text_{attack_name}_signature.png"  
                cv2.imwrite(attacked_signature_path, attacked_signature)

                attacked_extracted_message = image_to_text(attacked_signature_path)

                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(message, attacked_extracted_message, type="text"), 7),
                }
            
        print("\n")

    with open(f"{results_path}DWT_Text.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)

def dwt_img(dataset_folders, watermark_path, export_path="./dataset_export/", results_path="./dataset_results/", limit=200, signature_size=200, wavelet_level=1):
    print(f"\n\nAlgorithm DWT Image")
    
    # Metrics Sum
    metrics = {}

    # Attacked Metrics Sum
    attacked_metrics = {}
    
    for folder in dataset_folders:
        print(f"Working on Folder {folder}:")
        print(f"Progress form {limit} img: ", end=" ")

        cover_images_path = f"./dataset/{folder}/"

        for i in range(1, limit+1):
            print(i, end=" ")

            # Generate Paths
            cover_path = f"{cover_images_path}{folder}-{i}.png"
            stego_path = f"{export_path}{folder}-{i}_DWT_Image.png"
            signature_path = f"{export_path}{folder}-{i}_DWT_Image_signature.png"

            # Encode and Decode the text watermark
            cover_img = cv2.imread(cover_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

            dwt = DWT_Watermark()
            dwt.set_signature_size(signature_size)
            dwt.set_level(wavelet_level)
            
            stego_img = dwt.embed(cover_img, watermark_img)
            cv2.imwrite(stego_path, stego_img)

            signature = dwt.extract(stego_img)
            cv2.imwrite(signature_path, signature)

            ##### Calculate Metrics #####
            cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
            stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            metrics[stego_path] = {
                "MSE": mse(cover_img, stego_img),
                "SNR": snr(cover_img, stego_img),
                "PSNR": psnr(cover_img, stego_img),
                "SSIM": calculate_ssim(cover_img, stego_img),
                "NC": nc(watermark_img, signature),
            }

            ##### Calculate Attacked Metrics #####
            attacked_stego_images = apply_attacks(stego_img)

            attacked_metrics[stego_path] = {}
            for attack_name, attacked_stego_img in zip(attacked_stego_images.keys(), attacked_stego_images.values()):
                attacked_stego_path = f"{export_path}{folder}-{i}_DWT_Image_{attack_name}.png"  
                cv2.imwrite(attacked_stego_path, attacked_stego_img)

                # decode watermark from attacked stego image
                attacked_stego_image = cv2.imread(attacked_stego_path)
                attacked_signature = dwt.extract(attacked_stego_image)

                attacked_signature_path = f"{export_path}{folder}-{i}_DWT_Image_{attack_name}_signature.png"  
                cv2.imwrite(attacked_signature_path, attacked_signature)

                # Calculate
                attacked_stego_img = cv2.imread(attacked_stego_path, 0)
                attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))
                
                attacked_signature = cv2.imread(attacked_signature_path, 0)
                attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

                attacked_metrics[stego_path][attack_name] = {
                    "MSE": round(mse(cover_img, attacked_stego_img), 7),
                    "SNR": round(snr(cover_img, attacked_stego_img), 7),
                    "PSNR": round(psnr(cover_img, attacked_stego_img), 7),
                    "SSIM": round(calculate_ssim(cover_img, attacked_stego_img), 7),
                    "NC": round(nc(watermark_img, attacked_signature), 7),
                }
            
        print("\n")

    with open(f"{results_path}DWT_Image.json", 'w') as file:
        json.dump({
            "Normal": metrics,
            "Attacked": attacked_metrics
        }, file, indent=4)


def calc_result(json_path):
    file = open(json_path)
 
    data = json.load(file)

    result = {
        "Normal": {},
        "Attacked": {}}
    
    # normal matrices
    normal_size = len(data["Normal"].values())
    for matrices in data["Normal"].values():
        for key, value in matrices.items():
            if key in result["Normal"]:
                result["Normal"][key] += value / normal_size
            else:
                result["Normal"][key] = value / normal_size

    # attacked matrices
    attacked_size = len(data["Attacked"].values())
    for img_attacks in data["Attacked"].values():
        for attack_name, attack_matrices in img_attacks.items():
            if not attack_name in result["Attacked"]:
                result["Attacked"][attack_name] = {}

            for key, value in attack_matrices.items():
                if key in result["Attacked"][attack_name]:
                    result["Attacked"][attack_name][key] += abs(value / attacked_size)
                else:
                    result["Attacked"][attack_name][key] = value / attacked_size

    return result

def main():
    # 
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

    watermark_path = "./img/logo_1.jpg"
    dataset_folders = ["Normal", "COVID", "Lung_Opacity", "Viral Pneumonia"]    

    # 
    # lsb_text_rgb(dataset_folders, message)
    # lsb_text_gray(dataset_folders, message)
    # lsb_text_min_avr_max(dataset_folders, message)
    # dct_text(dataset_folders, message)
    # dwt_text(dataset_folders, message)

    # lsb_img_rgb(dataset_folders, watermark_path)
    # lsb_img_gray(dataset_folders, watermark_path)
    # lsb_img_min_avr_max(dataset_folders, watermark_path, signature_size=200, limit=10)
    # dct_img(dataset_folders, watermark_path)
    # dwt_img(dataset_folders, watermark_path)

    # Calculate Matrices
    # result = calc_result("./dataset_results/LSB_Text_RGB.json")
    # result = calc_result("./dataset_results/LSB_Text_Gray.json")
    # result = calc_result("./dataset_results/LSB_Text_Min_Avr_Max.json")

    # result = calc_result("./dataset_results/DCT_Text.json")
    # result = calc_result("./dataset_results/DCT_Image.json")

    # result = calc_result("./dataset_results/DWT_Text.json")
    result = calc_result("./dataset_results/DCT_Image.json")


if __name__ == "__main__":
    main()
