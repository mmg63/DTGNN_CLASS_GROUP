"""
OBSOLETE SCRIPT
This script loads the graph data and assigns multi-hot labels to each graph based on the color classes.
The labels are loaded from the label files and assigned to the graphs based on the filename match.
The processed data is saved as train_data.pt and test_data.pt files.

UPDATED on May 6th, 2025:
This script has been updated to load the graph data and assign one-hot labels to each graph based that whether the 
graph is cancerous or not.
The labels are loaded from the label files and assigned to the graphs based on the filename match.
The processed data is saved as train_data.pt and test_data.pt files.
The labels are now binary (0 or 1) instead of multi-hot.
"""
# ----------------------------------------------
#                  Imports
# ----------------------------------------------
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm


# ------------------------------------------------------------------
#              Define parent directory
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent    # The Base directory of the project is BIOMEDICAL_PROJECT
print(f'BASE directory: {BASE_DIR}')


# ----------------------------------------------
#                  Functions
# ----------------------------------------------

# Function to load and convert labels into a one-hot tensor (cancerous or not) (0:non-cancerous, 1:cancerous)
def load_label_one_hot(label_file_path, color_to_index):
    label_data = torch.load(label_file_path)  # Assumes format like: [("FILENAME", ["white", "blue"]), ...]
    label_dict = {}
    for i in range(len(label_data)):
        isCancerous = 0
        # Check if any of the colors are in the color_to_index mapping
        for color in label_data[i]['colors']:
            if color=='blue' or color=='yellow' or color=='red':
                isCancerous = 1
                break
        label_dict[label_data[i]['filename'].replace("mask_", "").replace("mask1_", "").replace(".png", "")] = isCancerous
                
    return label_dict


# Load Graphs and Assign Labels
# Load .pt graph files and attach .y (label) if matched
def load_graph_data(graph_folder, label_dict, append_filename=False):
    graph_folder = Path(graph_folder)
    data_list = []
    if append_filename:
        for graph_file in tqdm(graph_folder.glob("*.pt")):
            # Extract base filename without prefix/suffix
            stem = graph_file.stem.replace("_aug1.png", "").replace("_aug2.png", "").replace("_aug3.png", "").replace("_aug4.png", "").replace("_aug5.png", "").replace("_triangulation", "").replace(".jpg", "").replace("1024", "").replace("augmentedBenign", "").replace("_aug_0", "").replace("_aug_1", "").replace("_aug_2", "").replace("_aug_3", "")
        
            if stem in label_dict:
                data = torch.load(graph_file)
                if isinstance(data, Data):
                    data.y = label_dict[stem]
                    data_list.append([data, stem])
    else:
        for graph_file in tqdm(graph_folder.glob("*.pt")):
            # Extract base filename without prefix/suffix
            stem = graph_file.stem.replace(".jpg_triangulation", "")
            if stem in label_dict:
                data = torch.load(graph_file)
                if isinstance(data, Data):
                    data.y = label_dict[stem]
                    data_list.append(data)
    return data_list


# ----------------------------------------------
#                  Setup
# ----------------------------------------------
# To determine that the file name should be added to the graph data or not
APPEND_FILENAME = True

# Define paths
train_labels_files_path = f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/labels_train.pt'
test_labels_files_path = f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/labels_test_pathologist1.pt' 
graph_folder_foder_path = f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/1024'
final_dataset_path_to_save = f'{BASE_DIR}/model_1_DTGCN_Class/DTDataset_Class/Graphs/1024/FullDataset'

# Define label classes (colors)
color_classes = ['white', 'green', 'blue', 'yellow', 'red']  # Gleason Grade: [0:back ground, 1: benign, 3, 4, 5]
color_to_index = {color: i for i, color in enumerate(color_classes)}

# ------------------------------------------------
#                  Load Data
# ------------------------------------------------
# Load the labels for one-hot encoding (cancerous or not)
train_labels = load_label_one_hot(train_labels_files_path, color_to_index)
test_labels = load_label_one_hot(test_labels_files_path, color_to_index)

# labels_union = train_labels | test_labels


# ------------------------------------------------
#                  Save Trianglated datasets 
#                     with one-hot labels
# ------------------------------------------------
train_dataset_with_labels = load_graph_data(graph_folder_foder_path, train_labels, append_filename=APPEND_FILENAME)
test_dataset_with_labels = load_graph_data(graph_folder_foder_path, test_labels, append_filename=APPEND_FILENAME)

print(f"\n\n\n✅ Loaded {len(train_dataset_with_labels)} training graphs.\n")
print(f"\n\n\n✅ Loaded {len(test_dataset_with_labels)} testing graphs.\n")


# ------------------------------------------------
#                  Save the processed data
# ------------------------------------------------
# Create the directory if it doesn't exist
Path(final_dataset_path_to_save).mkdir(parents=True, exist_ok=True)
if APPEND_FILENAME:
    # Save the datasets with filenames
    torch.save(train_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TrainGraphAugmentedDatasetWithLabels_and_filenames_one_hot.pt")
    torch.save(test_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TestGraphAugmentedDatasetWithLabels_and_filenames_one_hot.pt")
    sample = train_dataset_with_labels[0][0]
    sample_filename = train_dataset_with_labels[0][1]
    print(f"Filename: {sample_filename}\n")
else:
    # Save the datasets without filenames
    torch.save(train_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TrainGraphDatasetWithLabels_one_hot.pt")
    torch.save(test_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TestGraphDatasetWithLabels_one_hot.pt")
    sample = train_dataset_with_labels[0]


# 4- preview a sample
print("🔍 Sample Graph Info\n")
print(f"Node features: {sample.x.shape}\n")
print(f"Edge index: {sample.edge_index.shape}\n")
print(f"Labels (one-hot): {sample.y}\n")

# Count the number of Benign and Cancerous graphs in the training dataset
tc, cc = 0, 0  #Total count, Cancerous count
for _,graph in enumerate(train_dataset_with_labels):
    tc += 1
    if graph[0].y == 1:
        cc +=1
print('\n\n\nThe number of graphs in the train dataset')
print('Total Graph = ', tc)
print('Total Cancerous graphs = ', cc)
print('Total benign graphs = ', tc - cc)


# Count the number of Benign and Cancerous graphs in the test dataset
tc, cc = 0, 0  #Total count, Cancerous count
for _,graph in enumerate(test_dataset_with_labels):
    tc += 1
    if graph[0].y == 1:
        cc +=1
print('\n\n\nThe number of graphs in the test dataset')
print('Total Graph = ', tc)
print('Total Cancerous graphs = ', cc)
print('Total benign graphs = ', tc - cc)

