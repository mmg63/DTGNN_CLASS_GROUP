
def imageLabNormalization(image_lab):
    """
    Normalize the LAB channels in the input image.
    Args:
        image_lab: Input image in LAB color space.
    Returns:
        Normalized LAB image.  
    """
    # Print raw LAB values before normalization
    # print("Raw LAB Min/Max Before Normalization:")
    # print("L:", image_lab[..., 0].min(), image_lab[..., 0].max())  # Expected: [0,255]
    # print("A:", image_lab[..., 1].min(), image_lab[..., 1].max())  # Expected: ~[0,255]
    # print("B:", image_lab[..., 2].min(), image_lab[..., 2].max())  # Expected: ~[0,255]

    # Apply normalization
    # Compute actual min/max for A and B
    L_min, L_max = 0, 255  # L always ranges from 0-255
    A_min, A_max = image_lab[..., 1].min(), image_lab[..., 1].max()
    B_min, B_max = image_lab[..., 2].min(), image_lab[..., 2].max()

    # Normalize directly on image_lab
    image_lab[..., 0] = image_lab[..., 0] / 255.0  # Normalize L to [0,1]
    image_lab[..., 1] = 2 * (image_lab[..., 1] - A_min) / (A_max - A_min) - 1  # Normalize A to [-1,1]
    image_lab[..., 2] = 2 * (image_lab[..., 2] - B_min) / (B_max - B_min) - 1  # Normalize B to [-1,1]

    # Print final LAB ranges to verify
    # print("Normalized LAB Min/Max:")
    # print("L:", image_lab[..., 0].min(), image_lab[..., 0].max())  # Should be [0,1]
    # print("A:", image_lab[..., 1].min(), image_lab[..., 1].max())  # Should be ~[-1,1]
    # print("B:", image_lab[..., 2].min(), image_lab[..., 2].max())  # Should be ~[-1,1] 
    
    return image_lab


def imageLabDenormalization(image_lab):
    """
    Denormalize the LAB channels in the input image.
    Args:
        image_lab: Input image in LAB color space.
    Returns:
        Denormalized LAB image.
    """
    # Print raw LAB values before normalization
    # print("Raw LAB Min/Max Before Normalization:")
    # print("L:", image_lab[..., 0].min(), image_lab[..., 0].max())  # Expected: [0,255]
    # print("A:", image_lab[..., 1].min(), image_lab[..., 1].max())  # Expected: ~[0,255]
    # print("B:", image_lab[..., 2].min(), image_lab[..., 2].max())  # Expected: ~[0,255]

    # Apply normalization
    # Compute actual min/max for A and B
    L_min, L_max = 0, 255  # L always ranges from 0-255
    A_min, A_max = image_lab[..., 1].min(), image_lab[..., 1].max()
    B_min, B_max = image_lab[..., 2].min(), image_lab[..., 2].max()

    # Normalize directly on image_lab
    image_lab[..., 0] = image_lab[..., 0] * 255.0  # Normalize L to [0,1]
    image_lab[..., 1] = ((image_lab[..., 1] + 1) / 2) * (A_max - A_min) + A_min  # Normalize A to [-1,1]
    image_lab[..., 2] = ((image_lab[..., 2] + 1) / 2) * (B_max - B_min) + B_min  # Normalize B to [-1,1]

    # Print final LAB ranges to verify
    # print("Normalized LAB Min/Max:")
    # print("L:", image_lab[..., 0].min(), image_lab[..., 0].max())  # Should be [0,1]
    # print("A:", image_lab[..., 1].min(), image_lab[..., 1].max())  # Should be ~[-1,1]
    # print("B:", image_lab[..., 2].min(), image_lab[..., 2].max())  # Should be ~[-1,1] 
    
    return image_lab


def imageZscoreNormalization(image):
    """
    Normalize the input image using Z-score normalization.
    Args:
        image: Input image.
    Returns:
        Z-score normalized image.
    """
    # Print raw image values before normalization
    # print("Raw Image Min/Max Before Normalization:")
    # print("Min:", image.min(), "Max:", image.max())

    # Apply Z-score normalization
    mean = image.mean()
    std = image.std()
    image_normalized = (image - mean) / std

    # Print final image ranges to verify
    # print("Z-score Normalized Min/Max:")
    # print("Min:", image_normalized.min(), "Max:", image_normalized.max())
    
    return image_normalized


def imageMinMaxNormalization(image):
    """
    Normalize the input image using min-max normalization.
    Args:
        image: Input image.
    Returns:
        Min-max normalized image.
    """
    # Print raw image values before normalization
    # print("Raw Image Min/Max Before Normalization:")
    # print("Min:", image.min(), "Max:", image.max())

    # Apply min-max normalization
    min_val = image.min()
    max_val = image.max()
    image_normalized = (image - min_val) / (max_val - min_val)

    # Print final image ranges to verify
    # print("Min-Max Normalized Min/Max:")
    # print("Min:", image_normalized.min(), "Max:", image_normalized.max())
    
    return image_normalized
