# Method to embed watermark into image
def inner_embed(self, B, signature):
    w, h = B.shape[:2]

    # Perform 4-level DWT with Haar wavelet
    LL, (HL, LH, HH) = pywt.dwt2(
        np.array(B[:32 * (w // 32), :32 * (h // 32)]), 'haar')
    LL_1, (HL_1, LH_1, HH_1) = pywt.dwt2(LL, 'haar')
    LL_2, (HL_2, LH_2, HH_2) = pywt.dwt2(LL_1, 'haar')
    LL_3, (HL_3, LH_3, HH_3) = pywt.dwt2(LL_2, 'haar')
    LL_4, (HL_4, LH_4, HH_4) = pywt.dwt2(LL_3, 'haar')
    bi_int_part, frac_part, combo_neg_idx, _ = self.__gene_embed_space(
        HH_2)  # Change this line

    # Generate embedding space for watermark
    HH_2 = self.__embed_sig(bi_int_part, frac_part,
                            combo_neg_idx, signature)  # And this line

    # Perform inverse DWT to reconstruct the image
    LL_3 = pywt.idwt2((LL_4, (HL_4, LH_4, HH_4)), 'haar')
    LL_2 = pywt.idwt2((LL_3, (HL_3, LH_3, HH_2)), 'haar')  # And this line
    LL_1 = pywt.idwt2((LL_2, (HL_2, LH_2, HH_1)), 'haar')
    LL = pywt.idwt2((LL_1, (HL_1, LH_1, HH)), 'haar')
    B[:32 * (w // 32), :32 * (h // 32)
      ] = pywt.idwt2((LL, (HL, LH, HH)), 'haar')

    return B
