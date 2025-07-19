# utils/training_utils.py
import argparse
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Biomedical Delaunay Graph Training.")
    parser.add_argument('--train_dataset_path', 
                        type=str, 
                        required=False, 
                        help='Path to the dataset file',
                        # default=f'/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Augmented_images_for_whole_dataset/Converted dataset/20250410_TrainGraphAugmentedDatasetWithLabels_and_filenames.pt')
                        default=f'/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Converted Dataset/20250406_TrainGraphDatasetWithLabels_and_filenames.pt')
    parser.add_argument('--test_dataset_path', 
                        type=str, 
                        required=False, 
                        help='Path to the Train dataset file',
                        # default=f'/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Augmented_images_for_whole_dataset/Converted dataset/20250410_TestGraphAugmentedDatasetWithLabels_and_filenames.pt')
                        default=f'/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/dataset/dataverse_files-2/Combined/Converted Dataset/20250406_TestGraphDatasetWithLabels_and_filenames.pt')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of labels/classes in the dataset')
    parser.add_argument('--k_fold', type=int, default=6, help='Number of k folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01)
    
    return parser.parse_args()

def get_device():
    # if torch.backends.mps.is_available():
    #     return torch.device('mps')
    # elif torch.cuda.is_available():
    #     return torch.device('cuda')
    # else:
    #     return torch.device('cpu')
    return torch.device('cpu') 

def compute_pos_weight(train_dataset, num_classes):
    label_matrix = torch.stack([graph[0].y for graph in train_dataset])
    pos_counts = label_matrix.sum(dim=0)
    neg_counts = len(train_dataset) - pos_counts
    return (neg_counts / (pos_counts + 1e-5)).to(torch.float32)