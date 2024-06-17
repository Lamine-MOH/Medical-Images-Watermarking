import tkinter as tk
from tkinter import filedialog
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
from PIL import ImageTk    
import os
import lsb as lsb
from metrics import *
from text_to_img import *
from attack import *
from dct_watermark import *
from dwt_watermark import *

# the function that handel showing the images
def show_image(image_path, image_preview, image_entry=None, thumbnail_size=(250, 250), image_info_label=None):
    if image_path == (): return  # Exit if get an empty image path

    img = Image.open(image_path) # open the image

    # get image size
    height, width = img.height, img.width

    # show the image
    img.thumbnail(thumbnail_size)  # Resize the image to fit within a 200x200 pixel box
    img = ImageTk.PhotoImage(img)
    image_preview.configure(image=img)
    image_preview.image = img  # Keep a reference to the image to prevent it from being garbage collected

    # Insert the image path into the entry widget
    if not image_entry == None:
        image_entry.delete(0, tk.END)  # Clear the entry widget
        image_entry.insert(tk.END, image_path)  # Insert the image path

    # show the image info
    if not image_info_label == None:
        img_size = f"{height}px * {width}px"
        image_info_label.configure(text=img_size)

# the function that handel hiding the images
def hide_image(image_label, image_info_label=None):
    # hide the image
    image_label.configure(image=None)
    image_label.image = None

    # hide the image info
    if not image_info_label == None:
        image_info_label.configure(text="")

# the function that handel calculating the available embedding space on the cover image
def calculate_embedding_space():
    # get the cover path
    cover_path = cover_image_entry.get()
    if cover_path == "": return

    # get cover size
    cover_img = Image.open(cover_path)
    height, width = cover_img.height, cover_img.width

    # LSB Algorithm
    if selected_algorithm.get() == "LSB":
        if selected_lsb.get() == "LSB RGB":
            embedding_space = height * width * 3
        elif selected_lsb.get() == "LSB Gray Scale":
            embedding_space = height * width
        elif selected_lsb.get() == "LSB Min Avr Max":
            embedding_space = height * width / 3 * 2

    # DCT Algorithm
    elif selected_algorithm.get() == "DCT":
        block_size = int(dct_block_size_entry.get())
        embedding_space = (height // block_size) * (width // block_size)

    # DWT Algorithm
    elif selected_algorithm.get() == "DWT":
        level = int(wavelet_level_entry.get())
        embedding_space = (height // (2**level)) * (width // (2**level))

    # show the embedding size
    embedding_space_label.config(text=f"Embedding Space: {embedding_space} bits =  {embedding_space/8} bytes")

# the function that handel calculating the required embedding space for the watermark
def calculate_required_embedding_space():
    # text watermark
    if change_wm_type_button["text"] == "Change to Image":
        message = watermark_entry.get()
        if message == "": return

        required_embedding_space = len(message) * 8

    # image watermark
    elif change_wm_type_button["text"] == "Change to Text":
        watermark_path = watermark_entry.get()
        if watermark_path == "": return

        signature_size = int(signature_size_entry.get())

        # LSB Algorithm
        if selected_algorithm.get() == "LSB":
            if selected_lsb.get() == "LSB RGB":
                required_embedding_space = signature_size * signature_size * 3
            elif selected_lsb.get() == "LSB Gray Scale":
                required_embedding_space = signature_size * signature_size
            elif selected_lsb.get() == "LSB Min Avr Max":
                required_embedding_space = signature_size * signature_size

        # DCT and DWT Algorithm
        elif selected_algorithm.get() == "DCT" or selected_algorithm.get() == "DWT":
            required_embedding_space = signature_size * signature_size

    # show the required embedding size
    required_embedding_space_label.config(text=f"Required Embedding Space: {required_embedding_space} bytes")

# the function that handel showing the cover image and the paths
def show_cover_image(image_path):
    if image_path == (): return  # Exit if get an empty image path

    reset_interface()

    # show cover image
    show_image(image_path, cover_image_preview, cover_image_entry, image_info_label=cover_image_info)

    # Insert the coded image path into the entry widget
    file_name, file_extension = os.path.splitext(image_path)
    stego_image_entry.delete(0, tk.END)  # Clear the entry widget
    stego_image_entry.insert(tk.END, f"{file_name}_{selected_algorithm.get()}{file_extension}")  # Insert the image path

    # Hide the stego image
    hide_image(stego_image_preview, stego_image_info)

    # Clear the decoded message
    decoded_message_label.config(text = "")

    # calculate the imbedding space
    calculate_embedding_space()

# the function that handel encoding the watermark on the cover image
def encode_watermark():
    # get paths
    cover_path = cover_image_entry.get()
    stego_path = stego_image_entry.get()
    signature_size = int(signature_size_entry.get())

    # LSB Algorithm
    if selected_algorithm.get() == "LSB":
        text = watermark_entry.get()
        
        # text watermark
        if change_wm_type_button["text"] == "Change to Image":
            # Encode the Message
            if selected_lsb.get() == "LSB RGB":
                lsb.encode_lsb_text_rgb(cover_path, text, stego_path)
            elif selected_lsb.get() == "LSB Gray Scale":
                lsb.encode_lsb_text_gray(cover_path, text, stego_path)
            elif selected_lsb.get() == "LSB Min Avr Max":
                lsb.encode_lsb_text_min_avr_max(cover_path, text, stego_path)

        # image watermark
        elif change_wm_type_button["text"] == "Change to Text":
            # Encode the watermark
            if selected_lsb.get() == "LSB RGB":
                lsb.encode_lsb_img_rgb(cover_path, text, stego_path, signature_size=signature_size)
            elif selected_lsb.get() == "LSB Gray Scale":
                lsb.encode_lsb_img_gray(cover_path, text, stego_path, signature_size=signature_size)
            elif selected_lsb.get() == "LSB Min Avr Max":
                lsb.encode_lsb_img_min_avr_max(cover_path, text, stego_path, signature_size=signature_size)

    # DCT Algorithm
    elif selected_algorithm.get() == "DCT":
        # text watermark
        if change_wm_type_button["text"] == "Change to Image":
            # generate binary image
            watermark_path = "./img/binary_img.png"
            text = watermark_entry.get()
            text_to_image(text, path=watermark_path, image_size=(signature_size, signature_size))

        # image watermark
        elif change_wm_type_button["text"] == "Change to Text":
            watermark_path = watermark_entry.get()

        # get paths
        cover_image = cv2.imread(cover_path)
        watermark_image = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # create dct object
        dct = DCT_Watermark()
        dct.set_signature_size(signature_size)
        dct.set_block_size(int(dct_block_size_entry.get()))
        
        # encode the watermark
        stego_image = dct.embed(cover_image, watermark_image)
        cv2.imwrite(stego_path, stego_image)

    # DWT Algorithm
    elif selected_algorithm.get() == "DWT":
        # Encode a message
        if change_wm_type_button["text"] == "Change to Image":
            # generate binary image
            watermark_path = "./img/binary_img.png"
            text = watermark_entry.get()
            text_to_image(text, path=watermark_path, image_size=(signature_size, signature_size))

        elif change_wm_type_button["text"] == "Change to Text":
            watermark_path = watermark_entry.get()

        # get paths
        cover_image = cv2.imread(cover_path)
        watermark_image = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # create dwt object
        dwt = DWT_Watermark()
        dwt.set_signature_size(signature_size)
        dwt.set_level(int(wavelet_level_entry.get()))
        dwt.set_sub_band(selected_sub_band.get())

        # encode the watermark
        stego_image = dwt.embed(cover_image, watermark_image)
        cv2.imwrite(stego_path, stego_image)

    # show the stego image
    show_image(stego_path, stego_image_preview, image_info_label=stego_image_info)

    # decode the watermark
    decode_watermark()

    # calculate metrics
    calc_metrics()

# the function that handel decoding the watermark from the stego image
def decode_watermark(attacked=False):
    # get paths
    if not attacked: stego_path = stego_image_entry.get()
    else: stego_path = attacked_stego_image_entry.get()

    signature_size = int(signature_size_entry.get())

    # generate signature path
    file_name, file_extension = os.path.splitext(stego_path)
    signature_path = f"{file_name}_signature{file_extension}"

    # write the signature path on the signature entry
    signature_image_entry.delete(0, tk.END)
    signature_image_entry.insert(tk.END, signature_path)

    # LSB Algorithm
    if selected_algorithm.get() == "LSB":
        # text signature
        if change_wm_type_button["text"] == "Change to Image":
            if selected_lsb.get() == "LSB RGB":
                message = lsb.decode_lsb_text_rgb(stego_path)
            elif selected_lsb.get() == "LSB Gray Scale":
                message = lsb.decode_lsb_text_gray(stego_path)
            elif selected_lsb.get() == "LSB Min Avr Max":
                message = lsb.decode_lsb_text_min_avr_max(stego_path)

        # image signature
        elif change_wm_type_button["text"] == "Change to Text":
            if selected_lsb.get() == "LSB RGB":
                lsb.decode_lsb_img_rgb(stego_path, signature_path, signature_size=signature_size)
            elif selected_lsb.get() == "LSB Gray Scale":
                lsb.decode_lsb_img_gray(stego_path, signature_path, signature_size=signature_size)
            elif selected_lsb.get() == "LSB Min Avr Max":
                lsb.decode_lsb_img_min_avr_max(stego_path, signature_path, signature_size=signature_size)

    # DCT Algorithm
    elif selected_algorithm.get() == "DCT":
        stego_image = cv2.imread(stego_path)
        
        # create dct object
        dct = DCT_Watermark()
        dct.set_signature_size(int(signature_size_entry.get()))
        dct.set_block_size(int(dct_block_size_entry.get()))

        # extract the signature
        signature = dct.extract(stego_image)
        cv2.imwrite(signature_path, signature)

    # DWT Algorithm
    elif selected_algorithm.get() == "DWT":
        stego_image = cv2.imread(stego_path)
        
        # create dwt object
        dwt = DWT_Watermark()
        dwt.set_signature_size(int(signature_size_entry.get()))
        dwt.set_level(int(wavelet_level_entry.get()))
        dwt.set_sub_band(selected_sub_band.get())

        # extract the signature
        signature = dwt.extract(stego_image)
        cv2.imwrite(signature_path, signature)

    # show the signature
    # text signature
    if change_wm_type_button["text"] == "Change to Image":
        # LSB algorithm
        if selected_algorithm.get() == "LSB":
            if not attacked: decoded_message_label.config(text = message)
            else: attacked_decoded_message_label.config(text = message)

        # DCT and DWT algorithm
        else:
            if not attacked: decoded_message_label.config(text = image_to_text(signature_path))
            else: attacked_decoded_message_label.config(text = image_to_text(signature_path))

    # image signature
    else:
        if not attacked: show_image(signature_path, signature_image_preview, thumbnail_size=(150, 150), image_info_label=signature_image_info)
        else: show_image(signature_path, attacked_signature_image_preview, thumbnail_size=(150, 150))

# the function that handel applying attacks on the stego image
def apply_attacks():
    # get paths
    stego_path = stego_image_entry.get()

    # generate paths
    file_name, file_extension = os.path.splitext(stego_path)
    attacked_stego_path = f"{file_name}_attacked{file_extension}"
    attacked_signature_path = f"{file_name}_attacked_signature{file_extension}"

    # show paths
    attacked_stego_image_entry.delete(0, tk.END)
    attacked_stego_image_entry.insert(tk.END, attacked_stego_path)

    attacked_signature_image_entry.delete(0, tk.END)
    attacked_signature_image_entry.insert(tk.END, attacked_signature_path)

    # apply attacks
    img = cv2.imread(stego_path, 0)

    if (crop_var.get() == 1):
        img = Attack.crop(img)
    
    if (resize_var.get() == 1):
        img = Attack.resize(img)
    
    if (scale_var.get() == 1):
        img = Attack.resize(img, width_resizing=2, height_resizing=1)
    
    if (shear_var.get() == 1):
        img = Attack.shear(img)
    
    if (salt_var.get() == 1):
        img = Attack.salt(img)
    
    if (pepper_var.get() == 1):
        img = Attack.pepper(img)
    
    if (gaussian_noise_var.get() == 1):
        img = Attack.gaussian_noise(img)
    
    if (rotate90_var.get() == 1):
        img = Attack.rotate90(img)
    
    if (rotate180_var.get() == 1):
        img = Attack.rotate180(img)
    
    if (histogram_equalization_var.get() == 1):
        img = Attack.histogram_equalization(img)
    
    if (randline_var.get() == 1):
        img = Attack.randline(img)
    
    if (cover_var.get() == 1):
        img = Attack.cover(img)
    
    if (compress_var.get() == 1):
        img = Attack.compress(img)
    
    if (quantize_var.get() == 1):
        img = Attack.quantize(img)
    
    if (blur_var.get() == 1):
        img = Attack.blur(img)
    
    if (median_filter_var.get() == 1):
        img = Attack.median_filter(img)
    

    # save the attacked stego image
    cv2.imwrite(attacked_stego_path, img)

    # show the attacked stego image
    show_image(attacked_stego_path, attacked_stego_image_preview)

    # extract the signature from the attacked stego image
    decode_watermark(attacked=True)
    # show_image(attacked_signature_path, attacked_signature_image_preview, thumbnail_size=(150, 150))

    # calculate attacked metrics
    calc_attacked_metrics()

# the function that handel calculating the watermarking metrics
def calc_metrics():
    # Get images path
    cover_path = cover_image_entry.get()
    stego_path = stego_image_entry.get()
    watermark_path = watermark_entry.get()
    signature_path = signature_image_entry.get()

    # check if not empty entry's
    if not (cover_path == "" or stego_path == ""):
        # Read two images
        cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        stego_img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

        # check if have same shapes
        if not cover_img.shape[0] == stego_img.shape[0] or not cover_img.shape[1] == stego_img.shape[1]:
            stego_img = cv2.resize(stego_img, (cover_img.shape[1], cover_img.shape[0]))

        # Calculate metrics
        MSE_result.config(text = round(mse(cover_img, stego_img), 7))
        SNR_result.config(text = round(snr(cover_img, stego_img), 7))
        PSNR_result.config(text = round(psnr(cover_img, stego_img), 7))
        SSIM_result.config(text = round(calculate_ssim(cover_img, stego_img), 7))

    # text signature
    if change_wm_type_button["text"] == "Change to Image":
        # get text
        original_message = watermark_entry.get()
        extracted_message = decoded_message_label.cget("text")

        NC_result.config(text = round(nc(original_message, extracted_message, type="text"), 7))

    # text signature
    else:
        # get paths
        watermark_path = watermark_entry.get()
        signature_path = signature_image_entry.get()

        # check if not empty entry's
        if not (watermark_path == "" or signature_path == ""):
            # Read two images
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)

            # check if have same shapes
            if not watermark.shape[0] == signature.shape[0] or not watermark.shape[1] == signature.shape[1]:
                signature = cv2.resize(signature, (watermark.shape[1], watermark.shape[0]))

            # Calculate metrics
            NC_result.config(text = round(nc(watermark, signature), 7))

# the function that handel calculating the attacked watermarking metrics
def calc_attacked_metrics():
    # Get images path
    cover_path = cover_image_entry.get()
    attacked_stego_path = attacked_stego_image_entry.get()
    
    # check if not empty entry's
    if not (cover_path == "" or attacked_stego_path == ""):
        # Read two images
        cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        attacked_stego_img = cv2.imread(attacked_stego_path, cv2.IMREAD_GRAYSCALE)

        # check if have same shapes
        if not cover_img.shape[0] == attacked_stego_img.shape[0] or not cover_img.shape[1] == attacked_stego_img.shape[1]:
            attacked_stego_img = cv2.resize(attacked_stego_img, (cover_img.shape[1], cover_img.shape[0]))

        # Calculate metrics
        attacked_SNR_result.config(text = round(snr(cover_img, attacked_stego_img), 7))
        attacked_MSE_result.config(text = round(mse(cover_img, attacked_stego_img), 7))
        attacked_PSNR_result.config(text = round(psnr(cover_img, attacked_stego_img), 7))
        attacked_SSIM_result.config(text = round(calculate_ssim(cover_img, attacked_stego_img), 7))

    # text signature
    if change_wm_type_button["text"] == "Change to Image":
        # get text
        original_message = watermark_entry.get()
        extracted_message = attacked_decoded_message_label.cget("text")

        attacked_NC_result.config(text = round(nc(original_message, extracted_message, type="text"), 7))

    # image signature
    else:
        # get paths
        watermark_path = watermark_entry.get()
        attacked_signature_path = attacked_signature_image_entry.get()

        # check if not empty entry's
        if not (watermark_path == "" or attacked_signature_path == ""):
            # Read two images
            watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            attacked_signature = cv2.imread(attacked_signature_path, cv2.IMREAD_GRAYSCALE)

            # check if have same shapes
            if not watermark.shape[0] == attacked_signature.shape[0] or not watermark.shape[1] == attacked_signature.shape[1]:
                attacked_signature = cv2.resize(attacked_signature, (watermark.shape[1], watermark.shape[0]))

            # Calculate metrics
            attacked_NC_result.config(text = round(nc(watermark, attacked_signature), 7))

# the function that handel resting the interface to the initial values
def reset_interface():
    # Clear the entry widget
    signature_size_entry.delete(0, tk.END)
    wavelet_level_entry.delete(0, tk.END)
    cover_image_entry.delete(0, tk.END)
    stego_image_entry.delete(0, tk.END)
    watermark_entry.delete(0, tk.END)
    signature_image_entry.delete(0, tk.END)
    attacked_stego_image_entry.delete(0, tk.END)
    attacked_signature_image_entry.delete(0, tk.END)

    # Reset the entry widget
    signature_size_entry.insert(tk.END, "200")
    wavelet_level_entry.insert(tk.END, "1")

    # Hide the images
    hide_image(cover_image_preview)
    hide_image(watermark_preview)
    hide_image(stego_image_preview)
    hide_image(signature_image_preview)
    hide_image(attacked_stego_image_preview)
    hide_image(attacked_signature_image_preview)

    # clear the images info
    cover_image_info.config(text="")
    watermark_info.config(text="")
    stego_image_info.config(text="")
    signature_image_info.config(text="")
    embedding_space_label.config(text="")

    # Clear the decoded message
    decoded_message_label.config(text = "")
    attacked_decoded_message_label.config(text = "")

    # Clear the Assessment values
    MSE_result.config(text = "0")
    SNR_result.config(text = "0")
    PSNR_result.config(text = "0")
    NC_result.config(text = "0")
    SSIM_result.config(text = "0")
    
    attacked_MSE_result.config(text = "0")
    attacked_SNR_result.config(text = "0")
    attacked_PSNR_result.config(text = "0")
    attacked_NC_result.config(text = "0")
    attacked_SSIM_result.config(text = "0")

# the function that handle the encoding algorithm selection
def algorithm_selected(algorithm):
    reset_interface()

    # change watermark type to text
    if change_wm_type_button["text"] == "Change to Text":
        change_watermark_type()

    # show options
    if algorithm == "LSB":
        lsb_menu.grid(row=0, column=0, padx=5)
        selected_lsb.set("LSB RGB")
        signature_size_frame.grid_forget()
        dct_block_size_frame.grid_forget()
        sub_band_menu.grid_forget()
        wavelet_level_frame.grid_forget()

    elif algorithm == "DCT":
        lsb_menu.grid_forget()
        signature_size_frame.grid(row=0, column=1, padx=5)
        dct_block_size_frame.grid(row=0, column=2, padx=5)
        sub_band_menu.grid_forget()
        wavelet_level_frame.grid_forget()

    elif algorithm == "DWT":
        lsb_menu.grid_forget()
        signature_size_frame.grid(row=0, column=1, padx=5)
        dct_block_size_frame.grid_forget()
        wavelet_level_frame.grid(row=0, column=3, padx=5)
        selected_sub_band.set("HH")
        sub_band_menu.grid(row=0, column=4, padx=5)

# the function that handle the encoding LSB algorithm selection
def lsb_selected(lsb_type):
    # calculate embedding space
    calculate_embedding_space()

    # calculate required embedding space
    calculate_required_embedding_space()

# the function that handle the sub band selection
def sub_band_selected(lsb_type):
    pass

# the function that handle changing the watermark type
def change_watermark_type():
    # clear watermark image
    watermark_entry.delete(0, tk.END)

    # clear required embedding space
    required_embedding_space_label.config(text="")

    # change watermark type to image
    if change_wm_type_button["text"] == "Change to Image":
        change_wm_type_button.config(text = "Change to Text")
        message_label.config(text = "Image:")

        # show options
        if selected_algorithm.get() == "LSB":
            signature_size_frame.grid(row=0, column=1, padx=5)

        # show watermark image 
        watermark_browse_button.grid(row=1, column=2)
        watermark_preview.grid(row=3, column=0)
        watermark_info.grid(row=4, column=0)

        # show signature image
        signature_image_entry.grid(row=0, column=0)
        signature_image_browse_button.grid(row=0, column=1)
        signature_image_preview.grid(row=2, column=0)
        signature_image_info.grid(row=3, column=0)
        attacked_signature_image_entry.grid(row=1, column=1)
        attacked_signature_image_preview.grid(row=2, column=1)

        # hide the signature text
        decoded_message_label.grid_forget()
        attacked_decoded_message_label.grid_forget()

    # change watermark type to text
    else:
        change_wm_type_button.config(text = "Change to Image")
        message_label.config(text = "Message:")

        # hide the image watermark
        hide_image(watermark_preview)
        watermark_info.config(text="")

        # hide options
        if selected_algorithm.get() == "LSB":
            signature_size_frame.grid_forget()

        # hide watermark image
        watermark_browse_button.grid_forget()
        watermark_preview.grid_forget()
        watermark_info.grid_forget()

        # hide signature image
        signature_image_entry.grid_forget()
        signature_image_browse_button.grid_forget()
        signature_image_preview.grid_forget()
        signature_image_info.grid_forget()
        attacked_signature_image_entry.grid_forget()
        attacked_signature_image_preview.grid_forget()

        # show the signature text
        decoded_message_label.grid(row=2, column=0)
        attacked_decoded_message_label.grid(row=2, column=1)


#####################################################################
root = tk.Tk()
root.title("Watermarking Module by Lamine")
root = tk.Frame(root)
root.pack(padx=10, pady=10)

# Title of the interface
tk.Label(root, text="Watermarking Module by Lamine", font=("Arial", 26)).grid(row=0, column=0, columnspan=4)

### The Options frame ###
options_frame = tk.Frame(root)
options_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

# Options 1
options_1_frame = tk.Frame(options_frame)
options_1_frame.grid(row=0, column=0)

# Button to reset the interface
reset_interface_button = tk.Button(options_1_frame, text="Reset", command=lambda: reset_interface())
reset_interface_button.grid(row=0, column=0, padx=5)

# Create a StringVar to store the selected encoding algorithm
selected_algorithm = StringVar(options_1_frame)
selected_algorithm.set("LSB")  # Default encoding algorithm

# Dropdown options for encoding algorithms
algorithm_options = ["LSB", "DCT", "DWT"]

# Create dropdown menu for selecting encoding algorithm
algorithm_menu = tk.OptionMenu(options_1_frame, selected_algorithm, *algorithm_options, command=algorithm_selected)
algorithm_menu.grid(row=0, column=1, padx=5)

# calculate embedding space button
embedding_space_button = tk.Button(options_1_frame, text="Embedding Space", command=calculate_embedding_space)
embedding_space_button.grid(row=0, column=2, padx=5) 

# calculate required embedding space button
required_embedding_space_button = tk.Button(options_1_frame, text="Required Embedding Space", command=calculate_required_embedding_space)
required_embedding_space_button.grid(row=0, column=3, padx=5) 

# Encode button
encode_button = tk.Button(options_1_frame, text="Encode", command=encode_watermark)
encode_button.grid(row=0, column=4, padx=5) 

# Decode button
decode_button = tk.Button(options_1_frame, text="Decode", command=decode_watermark)
decode_button.grid(row=0, column=5, padx=5)

# Options 2
options_2_frame = tk.Frame(options_frame)
options_2_frame.grid(row=1, column=0)

# LSB Technic
# Create a StringVar to store the LSB Technic
selected_lsb = StringVar(options_2_frame)
selected_lsb.set("LSB RGB")  # Default encoding algorithm
# Dropdown options for encoding algorithms
lsb_options = ["LSB RGB", "LSB Gray Scale", "LSB Min Avr Max"]
# Create dropdown menu for selecting encoding algorithm
lsb_menu = tk.OptionMenu(options_2_frame, selected_lsb, *lsb_options, command=lsb_selected)
lsb_menu.grid(row=0, column=0, padx=5)

# Signature Size
signature_size_frame = tk.Frame(options_2_frame)
signature_size_label = tk.Label(signature_size_frame, text="Signature Size:")
signature_size_label.grid(row=0, column=0)
signature_size_entry = tk.Entry(signature_size_frame, width=10)
signature_size_entry.insert(tk.END, "200")
signature_size_entry.grid(row=0, column=1)

# DCT block size
dct_block_size_frame = tk.Frame(options_2_frame)
dct_block_size_label = tk.Label(dct_block_size_frame, text="Block Size:")
dct_block_size_label.grid(row=0, column=0)
dct_block_size_entry = tk.Entry(dct_block_size_frame, width=10)
dct_block_size_entry.insert(tk.END, "2")
dct_block_size_entry.grid(row=0, column=1)

# DWT Wavelet level
wavelet_level_frame = tk.Frame(options_2_frame)
wavelet_level_label = tk.Label(wavelet_level_frame, text="Wavelet Level:")
wavelet_level_label.grid(row=0, column=0)
wavelet_level_entry = tk.Entry(wavelet_level_frame, width=10)
wavelet_level_entry.insert(tk.END, "1")
wavelet_level_entry.grid(row=0, column=1)

# DWT Sub band
# Create a StringVar to store the sub band options
selected_sub_band = StringVar(options_2_frame)
selected_sub_band.set("HH")  # Default sub band
# Dropdown options for sub bands
sub_band_options = ["LL", "HL", "LH", "HH"]
# Create dropdown menu for selecting sub band
sub_band_menu = tk.OptionMenu(options_2_frame, selected_sub_band, *sub_band_options, command=sub_band_selected)

### The Watermarking Frame ###
watermarking_frame = tk.Frame(root)
watermarking_frame.grid(row=2, column=0, columnspan=4)

# Cover Image
cover_frame = tk.Frame(watermarking_frame, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
cover_frame.grid(row=0, column=0, padx=5, pady=5)

cover_image_label = tk.Label(cover_frame, text="Cover Image", font=("Arial", 24))
cover_image_label.grid(row=0, column=0)

cover_image_entry_frame = tk.Frame(cover_frame)
cover_image_entry_frame.grid(row=1, column=0)

cover_image_entry = tk.Entry(cover_image_entry_frame, width=30)
cover_image_entry.grid(row=0, column=0)

cover_image_browse_button = tk.Button(cover_image_entry_frame, text="Browse", command=lambda: show_cover_image(filedialog.askopenfilename()))
cover_image_browse_button.grid(row=0, column=1)

cover_image_preview = tk.Label(cover_frame)
cover_image_preview.grid(row=2, column=0)

cover_image_info = tk.Label(cover_frame)
cover_image_info.grid(row=3, column=0)

embedding_space_label = tk.Label(cover_frame)
embedding_space_label.grid(row=4, column=0)


# Watermark Image
watermark_frame = tk.Frame(watermarking_frame, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
watermark_frame.grid(row=0, column=1, padx=5, pady=5)

watermark_image_label = tk.Label(watermark_frame, text="Watermark", font=("Arial", 24))
watermark_image_label.grid(row=0, column=0)

watermark_image_entry_frame = tk.Frame(watermark_frame)
watermark_image_entry_frame.grid(row=2, column=0)

change_wm_type_button = tk.Button(watermark_image_entry_frame, text="Change to Image", command=change_watermark_type)
change_wm_type_button.grid(row=0, column=1) 

message_label = tk.Label(watermark_image_entry_frame, text="Message:", font=("Arial", 16))
message_label.grid(row=1, column=0)

watermark_entry = tk.Entry(watermark_image_entry_frame, width=30)
watermark_entry.grid(row=1, column=1)  

watermark_browse_button = tk.Button(watermark_image_entry_frame, text="Browse", command=lambda: show_image(filedialog.askopenfilename(), watermark_preview, watermark_entry, thumbnail_size=(150, 150), image_info_label=watermark_info))

watermark_preview = tk.Label(watermark_frame)

watermark_info = tk.Label(watermark_frame)

required_embedding_space_label = tk.Label(watermark_frame)
required_embedding_space_label.grid(row=5, column=0)


# Stego Image
stego_frame = tk.Frame(watermarking_frame, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
stego_frame.grid(row=0, column=2, padx=5, pady=5)

cover_image_label = tk.Label(stego_frame, text="Watermarked Image", font=("Arial", 24))
cover_image_label.grid(row=0, column=0)

stego_image_entry_frame = tk.Frame(stego_frame)
stego_image_entry_frame.grid(row=1, column=0)

stego_image_entry = tk.Entry(stego_image_entry_frame, width=30)
stego_image_entry.grid(row=0, column=0)

stego_image_browse_button = tk.Button(stego_image_entry_frame, text="Browse", command=lambda: show_image(filedialog.askopenfilename(), stego_image_preview, stego_image_entry))
stego_image_browse_button.grid(row=0, column=1)

stego_image_preview = tk.Label(stego_frame)
stego_image_preview.grid(row=2, column=0)

stego_image_info = tk.Label(stego_frame)
stego_image_info.grid(row=3, column=0)

# Signature Image
signature_frame = tk.Frame(watermarking_frame, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
signature_frame.grid(row=0, column=3, padx=5, pady=5)

signature_label = tk.Label(signature_frame, text="Extracted Watermark", font=("Arial", 24))
signature_label.grid(row=0, column=0)

signature_image_frame = tk.Frame(signature_frame)
signature_image_frame.grid(row=1, column=0)

signature_image_entry = tk.Entry(signature_image_frame, width=30)
signature_image_browse_button = tk.Button(signature_image_frame, text="Browse", command=lambda: show_image(filedialog.askopenfilename(), signature_image_preview, signature_image_entry, thumbnail_size=(150, 150), image_info_label=signature_image_info))

signature_image_preview = tk.Label(signature_frame)

signature_image_info = tk.Label(signature_frame)

decoded_message_label = tk.Label(signature_frame, text="")
decoded_message_label.grid(row=2, column=0)

#########################################################################

### Assessment Results ###
assessment_frame = tk.Frame(root, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
assessment_frame.grid(row=3, column=0, padx=5, pady=5)

assessment_label = tk.Label(assessment_frame, text="Assessment of\nthe Watermarking", font=("Arial", 18))
assessment_label.grid(row=0, column=0)

calculate_quality_button = tk.Button(assessment_frame, text="Calculate", command=lambda: calc_metrics())
calculate_quality_button.grid(row=1, column=0, pady=20)

frame = tk.Frame(assessment_frame)
frame.grid(row=2, column=0)
MSE_label = tk.Label(frame, text="MSE:", font=("Arial", 16))
MSE_label.grid(row=0, column=0, padx=(0, 10))
MSE_result = tk.Label(frame, text="0", font=("Arial", 16))
MSE_result.grid(row=0, column=1)

frame = tk.Frame(assessment_frame)
frame.grid(row=3, column=0)
SNR_label = tk.Label(frame, text="SNR:", font=("Arial", 16))
SNR_label.grid(row=0, column=0, padx=(0, 10))
SNR_result = tk.Label(frame, text="0", font=("Arial", 16))
SNR_result.grid(row=0, column=1)

frame = tk.Frame(assessment_frame)
frame.grid(row=4, column=0)
PSNR_label = tk.Label(frame, text="PSNR:", font=("Arial", 16))
PSNR_label.grid(row=0, column=0, padx=(0, 10))
PSNR_result = tk.Label(frame, text="0", font=("Arial", 16))
PSNR_result.grid(row=0, column=1)

frame = tk.Frame(assessment_frame)
frame.grid(row=5, column=0)
NC_label = tk.Label(frame, text="NC:", font=("Arial", 16))
NC_label.grid(row=0, column=0, padx=(0, 10))
NC_result = tk.Label(frame, text="0", font=("Arial", 16))
NC_result.grid(row=0, column=1)

frame = tk.Frame(assessment_frame)
frame.grid(row=6, column=0)
SSIM_label = tk.Label(frame, text="SSIM:", font=("Arial", 16))
SSIM_label.grid(row=0, column=0, padx=(0, 10))
SSIM_result = tk.Label(frame, text="0", font=("Arial", 16))
SSIM_result.grid(row=0, column=1)


### Attacks ###
attacks_frame = tk.Frame(root, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
attacks_frame.grid(row=3, column=1, padx=5, pady=5)

assessment_label = tk.Label(attacks_frame, text="Attacks", font=("Arial", 24))
assessment_label.grid(row=0, column=0)

select_attack_frame = tk.Frame(attacks_frame)
select_attack_frame.grid(row=1, column=0)


crop_var = tk.IntVar()
crop_checkbox = tk.Checkbutton(select_attack_frame, text='Crop', font=("Arial", 14), variable=crop_var, onvalue=1, offvalue=0)
crop_checkbox.grid(row=0, column=0, sticky="w")

resize_var = tk.IntVar()
resize_checkbox = tk.Checkbutton(select_attack_frame, text='Resize', font=("Arial", 14), variable=resize_var, onvalue=1, offvalue=0)
resize_checkbox.grid(row=1, column=0, sticky="w")

scale_var = tk.IntVar()
scale_checkbox = tk.Checkbutton(select_attack_frame, text='Scale', font=("Arial", 14), variable=scale_var, onvalue=1, offvalue=0)
scale_checkbox.grid(row=2, column=0, sticky="w")

shear_var = tk.IntVar()
shear_checkbox = tk.Checkbutton(select_attack_frame, text='Shear', font=("Arial", 14), variable=shear_var, onvalue=1, offvalue=0)
shear_checkbox.grid(row=3, column=0, sticky="w")

salt_var = tk.IntVar()
salt_checkbox = tk.Checkbutton(select_attack_frame, text='Salt', font=("Arial", 14), variable=salt_var, onvalue=1, offvalue=0)
salt_checkbox.grid(row=4, column=0, sticky="w")

pepper_var = tk.IntVar()
pepper_checkbox = tk.Checkbutton(select_attack_frame, text='Pepper', font=("Arial", 14), variable=pepper_var, onvalue=1, offvalue=0)
pepper_checkbox.grid(row=5, column=0, sticky="w")

gaussian_noise_var = tk.IntVar()
gaussian_noise_checkbox = tk.Checkbutton(select_attack_frame, text='Gaussian Noise', font=("Arial", 14), variable=gaussian_noise_var, onvalue=1, offvalue=0)
gaussian_noise_checkbox.grid(row=6, column=0, sticky="w")

rotate90_var = tk.IntVar()
rotate90_checkbox = tk.Checkbutton(select_attack_frame, text='Rotate90', font=("Arial", 14), variable=rotate90_var, onvalue=1, offvalue=0)
rotate90_checkbox.grid(row=7, column=0, sticky="w")

rotate180_var = tk.IntVar()
rotate180_checkbox = tk.Checkbutton(select_attack_frame, text='Rotate180', font=("Arial", 14), variable=rotate180_var, onvalue=1, offvalue=0)
rotate180_checkbox.grid(row=0, column=1, sticky="w")

histogram_equalization_var = tk.IntVar()
histogram_equalization_checkbox = tk.Checkbutton(select_attack_frame, text='Histogram Equalization', font=("Arial", 14), variable=histogram_equalization_var, onvalue=1, offvalue=0)
histogram_equalization_checkbox.grid(row=1, column=1, sticky="w")

randline_var = tk.IntVar()
randline_checkbox = tk.Checkbutton(select_attack_frame, text='Randline', font=("Arial", 14), variable=randline_var, onvalue=1, offvalue=0)
randline_checkbox.grid(row=2, column=1, sticky="w")

cover_var = tk.IntVar()
cover_checkbox = tk.Checkbutton(select_attack_frame, text='Cover', font=("Arial", 14), variable=cover_var, onvalue=1, offvalue=0)
cover_checkbox.grid(row=3, column=1, sticky="w")

compress_var = tk.IntVar()
compress_checkbox = tk.Checkbutton(select_attack_frame, text='Compress', font=("Arial", 14), variable=compress_var, onvalue=1, offvalue=0)
compress_checkbox.grid(row=4, column=1, sticky="w")

quantize_var = tk.IntVar()
quantize_checkbox = tk.Checkbutton(select_attack_frame, text='Quantize', font=("Arial", 14), variable=quantize_var, onvalue=1, offvalue=0)
quantize_checkbox.grid(row=5, column=1, sticky="w")

blur_var = tk.IntVar()
blur_checkbox = tk.Checkbutton(select_attack_frame, text='Blur', font=("Arial", 14), variable=blur_var, onvalue=1, offvalue=0)
blur_checkbox.grid(row=6, column=1, sticky="w")

median_filter_var = tk.IntVar()
median_filter_checkbox = tk.Checkbutton(select_attack_frame, text='Median Filter', font=("Arial", 14), variable=median_filter_var, onvalue=1, offvalue=0)
median_filter_checkbox.grid(row=7, column=1, sticky="w")

apply_attack_button = tk.Button(attacks_frame, text="Apply Attacks", command=lambda: apply_attacks())
apply_attack_button.grid(row=2, column=0)


### Attacked Image ###
attacked_image_frame = tk.Frame(root, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
attacked_image_frame.grid(row=3, column=2, padx=5, pady=5)

attacked_stego_image_label = tk.Label(attacked_image_frame, text="Attacked Watermarked Image", font=("Arial", 14))
attacked_stego_image_label.grid(row=0, column=0)

attacked_stego_image_entry = tk.Entry(attacked_image_frame, width=30)
attacked_stego_image_entry.grid(row=1, column=0)

attacked_stego_image_preview = tk.Label(attacked_image_frame)
attacked_stego_image_preview.grid(row=2, column=0)

attacked_signature_image_label = tk.Label(attacked_image_frame, text="Attacked Extracted Watermark", font=("Arial", 14))
attacked_signature_image_label.grid(row=0, column=1)

attacked_signature_image_entry = tk.Entry(attacked_image_frame, width=30)

attacked_signature_image_preview = tk.Label(attacked_image_frame)

attacked_decoded_message_label = tk.Label(attacked_image_frame, text="")
attacked_decoded_message_label.grid(row=2, column=1)


### Attacked Assessment Results ###
attacked_assessment_frame = tk.Frame(root, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
attacked_assessment_frame.grid(row=3, column=3, padx=5, pady=5)

attacked_assessment_label = tk.Label(attacked_assessment_frame, text="Assessment of\nAttacked Watermarking", font=("Arial", 18))
attacked_assessment_label.grid(row=0, column=0)

calculate_attacked_quality_button = tk.Button(attacked_assessment_frame, text="Calculate", command=lambda: calc_attacked_metrics())
calculate_attacked_quality_button.grid(row=1, column=0, pady=20)

frame = tk.Frame(attacked_assessment_frame)
frame.grid(row=2, column=0)
attacked_MSE_label = tk.Label(frame, text="MSE:", font=("Arial", 16))
attacked_MSE_label.grid(row=0, column=0, padx=(0, 10))
attacked_MSE_result = tk.Label(frame, text="0", font=("Arial", 16))
attacked_MSE_result.grid(row=0, column=1)

frame = tk.Frame(attacked_assessment_frame)
frame.grid(row=3, column=0)
attacked_SNR_label = tk.Label(frame, text="SNR:", font=("Arial", 16))
attacked_SNR_label.grid(row=0, column=0, padx=(0, 10))
attacked_SNR_result = tk.Label(frame, text="0", font=("Arial", 16))
attacked_SNR_result.grid(row=0, column=1)

frame = tk.Frame(attacked_assessment_frame)
frame.grid(row=4, column=0)
attacked_PSNR_label = tk.Label(frame, text="PSNR:", font=("Arial", 16))
attacked_PSNR_label.grid(row=0, column=0, padx=(0, 10))
attacked_PSNR_result = tk.Label(frame, text="0", font=("Arial", 16))
attacked_PSNR_result.grid(row=0, column=1)

frame = tk.Frame(attacked_assessment_frame)
frame.grid(row=5, column=0)
attacked_NC_label = tk.Label(frame, text="NC:", font=("Arial", 16))
attacked_NC_label.grid(row=0, column=0, padx=(0, 10))
attacked_NC_result = tk.Label(frame, text="0", font=("Arial", 16))
attacked_NC_result.grid(row=0, column=1)

frame = tk.Frame(attacked_assessment_frame)
frame.grid(row=6, column=0)
attacked_SSIM_label = tk.Label(frame, text="SSIM:", font=("Arial", 16))
attacked_SSIM_label.grid(row=0, column=0, padx=(0, 10))
attacked_SSIM_result = tk.Label(frame, text="0", font=("Arial", 16))
attacked_SSIM_result.grid(row=0, column=1)


####
root.mainloop()
