import os
import cv2
from multiprocessing import Pool, cpu_count
from Main_Vor_Tess_v8_001_FN_20250111 import voronoi_tessellation_fn
from utils import imageMinMaxNormalization
import torch
from tqdm import tqdm
import time  # ✅ For timing
import argparse

# ------------------------------------------------------------------
#              argparse parameters
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Voronoi Tessellation on Images")
parser.add_argument('--folder_path', type=str, required=True,
                    default='/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Combined_images/',
                    help='The folder Path to the folder containing images')
parser.add_argument('--output_folder_tri', type=str, required=True,
                    default='/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Dataset_512/Triangulations/',
                    help='Destination folder for triangulations')
parser.add_argument('--output_folder_label', type=str, required=True, 
                    default='/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Dataset_512/Label Matrices/',
                    help='Destination folder for label matrices')
parser.add_argument('--NUM_GP', type=int, default=1024, help='Number of Voronoi points')
parser.add_argument('--compactness', type=float, default=0.5, help='Compactness parameter for Voronoi tessellation')
parser.add_argument('--doRGB2LAB', type=bool, default=False, help='Convert RGB to LAB color space')

args = parser.parse_args()


# ------------------------------------------------------------------
#              Voronoi tessellation parameters 
# ------------------------------------------------------------------
NUM_GP = args.NUM_GP
compactness = args.compactness
doRGB2LAB = args.doRGB2LAB
folder_path = args.folder_path
output_folder_tri = args.output_folder_tri
output_folder_label = args.output_folder_label

# ------------------------------------------------------------------
#                process_image function
# ------------------------------------------------------------------
def process_image(file_path):
    start_time = time.time()  # ✅ Start time for this image

    filename = os.path.basename(file_path)
    image = cv2.imread(file_path)
    if image is None:
        return (filename, None, None, 0.0)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_lab = imageMinMaxNormalization(image_lab)

    try:
        tri, label_matrix = voronoi_tessellation_fn(image_lab, k=NUM_GP, compactness=compactness, doRGBtoLAB=doRGB2LAB)
    except Exception as e:
        print(f"Error in triangulating {filename}: {e}")
        return (filename, None, None, 0.0)

    duration = time.time() - start_time  # ✅ Time taken
    return (filename, tri, label_matrix, duration)

# ------------------------------------------------------------------
#             get_image_files_in_folder function
# ------------------------------------------------------------------
def get_image_files_in_folder(folder_path):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]

# ------------------------------------------------------------------
#                 Main function
# ------------------------------------------------------------------
if __name__ == "__main__":
    image_files = get_image_files_in_folder(folder_path)
    num_processes = cpu_count()

    start_all = time.time()  # ✅ Start overall timer
    results = []
    total_duration = 0.0

    with Pool(processes=num_processes) as pool:
        for filename, tri, label_matrix, duration in tqdm(
                pool.imap_unordered(process_image, image_files),
                total=len(image_files),
                desc="Processing Images",
                dynamic_ncols=True):
            results.append((filename, tri, label_matrix))
            total_duration += duration

    for filename, tri, label_matrix in results:
        if tri is None:
            print(f"Triangulation for {filename} failed or image could not be read.")
        else:
            print(f"File: {filename} -> Triangulation has {len(tri.x)} triangles.")
            torch.save(tri, f"{output_folder_tri}{filename}_triangulation.pt")
            torch.save(label_matrix, f"{output_folder_label}{filename}_label_matrix.pt")

    end_all = time.time()
    avg_time = total_duration / len(results)
    print(f"\n✅ Average processing time per image: {avg_time:.2f} seconds")
    print(f"⏱️ Total processing time: {end_all - start_all:.2f} seconds")
    print("🎉 All done!")
