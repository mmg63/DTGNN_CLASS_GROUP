# ------------------------------------------------------------------
#              Import necessary libraries 
# ------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import torch
from tqdm import tqdm

# ------------------------------------------------------------------
#              Define parent directory
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f'BASE directory: {BASE_DIR}')


# ------------------------------------------------------------------
#              Define constants
# ------------------------------------------------------------------
FOLDER_PATH_TRAIN = f'{BASE_DIR}/dataset/Gleason_masks_train'
FOLDER_PATH_TEST_PATHOLOGIST_1 = f'{BASE_DIR}/dataset/Gleason_masks_test_pathologist1'
FOLDER_PATH_TEST_PATHOLOGIST_2 = f'{BASE_DIR}/dataset/Gleason_masks_test_pathologist2'

COLOR_MAP = {
    (255, 255, 255): 'white',
    (0,   255,   0): 'green',
    (0,   0,   255): 'blue',
    (255, 255,   0): 'yellow',
    (255, 0,     0): 'red'
}

DOWNSAMPLE_FACTOR = 8

# ------------------------------------------------------------------
#              Define a function to process each image
#              and extract colors from the image
#                   using multiprocessing
# ------------------------------------------------------------------
def process_image(folder_path, filename):
    try:
        filepath = os.path.join(folder_path, filename)
        
        with Image.open(filepath) as img:
            new_w = img.width  // DOWNSAMPLE_FACTOR
            new_h = img.height // DOWNSAMPLE_FACTOR
            if new_w < 1 or new_h < 1:
                small_img = img
            else:
                small_img = img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

            if small_img.mode != 'RGB':
                small_img = small_img.convert('RGB')
            
            arr = np.array(small_img)
            unique_pixels = np.unique(arr.reshape(-1, 3), axis=0)
            
            found_colors = set()
            for pix in unique_pixels:
                pix_tuple = tuple(pix)
                if pix_tuple in COLOR_MAP:
                    found_colors.add(COLOR_MAP[pix_tuple])
        
        return {'filename': filename, 'colors': list(found_colors)}

    except Exception as e:
        return {'filename': filename, 'colors': [], 'error': str(e)}

# ------------------------------------------------------------------
#              Function to run in parallel with tqdm
# ------------------------------------------------------------------
def run_in_parallel(folder_path):
    filenames = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = executor.map(process_image, repeat(folder_path), filenames)
        for result in tqdm(futures, total=len(filenames), desc="Processing images"):
            results.append(result)

    return results

# ------------------------------------------------------------------
#              Main block to execute the parallel run
# ------------------------------------------------------------------
if __name__ == '__main__':
    labels_train = run_in_parallel(FOLDER_PATH_TRAIN)
    labels_test_pathologist_1 = run_in_parallel(FOLDER_PATH_TEST_PATHOLOGIST_1)
    labels_test_pathologist_2 = run_in_parallel(FOLDER_PATH_TEST_PATHOLOGIST_2)

    for entry in labels_train:
        print(entry['filename'], entry['colors'])

    print(f"Total train files processed: {len(labels_train)}")
    print(f"Total test files processed (Pathologist 1): {len(labels_test_pathologist_1)}")
    print(f"Total test files processed (Pathologist 2): {len(labels_test_pathologist_2)}")

    # Save the results to .pt files
    torch.save(labels_train,
        f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/labels_train.pt')
    torch.save(labels_test_pathologist_1,
        f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/labels_test_pathologist1.pt')
    torch.save(labels_test_pathologist_2,
        f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/labels_test_pathologist2.pt')
    print('✅ Labels saved successfully!')
    print('✅ All files processed successfully!')

    print('✅ Done!')
