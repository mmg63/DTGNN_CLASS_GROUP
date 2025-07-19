"""
This script loads the graph data and assigns multi-hot labels to each graph based on the color classes.
The labels are loaded from the label files and assigned to the graphs based on the filename match.
The processed data is saved as train_data.pt and test_data.pt files.
"""
# ----------------------------------------------
#                  Imports
# ----------------------------------------------
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------
#                  Functions
# ----------------------------------------------
# Function to load and convert labels into a dictionary {filename: multi-hot tensor}
def load_label_dict(label_file_path, color_to_index):
    label_data = torch.load(label_file_path)  # Assumes format like: [("FILENAME", ["white", "blue"]), ...]
    label_dict = {}
    for i in range(len(label_data)):
        multi_hot = torch.zeros(len(color_to_index))
        for color in label_data[i]['colors']:
            if color in color_to_index:
                multi_hot[color_to_index[color]] = 1
        label_dict[label_data[i]['filename'].replace("mask_", "").replace("mask1_", "").replace(".png", "")] = multi_hot
    return label_dict


# Load Graphs and Assign Labels
# Load .pt graph files and attach .y (label) if matched
def load_graph_data(graph_folder, label_dict, append_filename=False):
    graph_folder = Path(graph_folder)
    data_list = []
    if append_filename:
        for graph_file in tqdm(graph_folder.glob("*.pt")):
            # Extract base filename without prefix/suffix
            stem = graph_file.stem.replace("_aug1.png", "").replace("_aug2.png", "").replace("_aug3.png", "").replace("_aug4.png", "").replace("_aug5.png", "").replace("_triangulation", "")
        
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
train_labels_files_path = '/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Labels/labels_train.pt'
test_labels_files_path = '/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Labels/labels_test_pathologist1.pt' 
graph_folder_foder_path = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Augmented_images_for_whole_dataset/Triangulations"
final_dataset_path_to_save = "/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/20250109-BioImages_VoronoiDiagramSimplified_FN/dataset/dataverse_files-2/Combined/Augmented_images_for_whole_dataset/Converted dataset/"

# Define label classes (colors)
color_classes = ['white', 'green', 'blue', 'yellow', 'red']
color_to_index = {color: i for i, color in enumerate(color_classes)}

# ------------------------------------------------
#                  Load Data
# ------------------------------------------------
# Load both label sets
train_labels = load_label_dict(train_labels_files_path, color_to_index)
test_labels = load_label_dict(test_labels_files_path, color_to_index)

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
    torch.save(train_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TrainGraphAugmentedDatasetWithLabels_and_filenames.pt")
    torch.save(test_dataset_with_labels, f"{final_dataset_path_to_save}/20250410_TestGraphAugmentedDatasetWithLabels_and_filenames.pt")
    sample = train_dataset_with_labels[0][0]
    sample_filename = train_dataset_with_labels[0][1]
    print(f"Filename: {sample_filename}\n")
else:
    # Save the datasets without filenames
    torch.save(train_dataset_with_labels, f"{final_dataset_path_to_save}/20250406_TrainGraphDatasetWithLabels.pt")
    torch.save(test_dataset_with_labels, f"{final_dataset_path_to_save}/20250406_TestGraphDatasetWithLabels.pt")
    sample = train_dataset_with_labels[0]


# 4- preview a sample
print("🔍 Sample Graph Info\n")
print(f"Node features: {sample.x.shape}\n")
print(f"Edge index: {sample.edge_index.shape}\n")
print(f"Labels (multi-hot): {sample.y}\n")
