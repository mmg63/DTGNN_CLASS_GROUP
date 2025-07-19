
"""
20250209-BioImages_Voronoi Diagram Simplified_FN
Updated version of the Voronoi Tessellation code. The code is simplified and the unnecessary parts are removed

"""
import numpy as np
import torch
import math, time
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops
import matplotlib.pyplot as plt
from _snic.lib import SNIC_main, Delaunay_triangulation
from utils import plot_nx_with_pos, plot_pyg_data_with_pos
from cffi import FFI
import torch.nn.functional as F
from timeit import default_timer as timer
import os, cv2
from datetime import datetime
from utils import plot_boundary_3d, plot_3d# For plotting 3d Matrices
from utils import imageLabDenormalization, imageLabNormalization, imageMinMaxNormalization, imageZscoreNormalization, drawBoundaries

# plt.ion()
# plt.ioff()

def plot_vertices(coords: list):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]

    # Plot the coordinates
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='red', marker='o')

    # Optionally, set the axis limits if you want to match the image size
    plt.xlim(0, 28)  # Example limits, change according to your image size
    plt.ylim(0, 28)

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Coordinates')

    # Show grid for better visibility
    plt.grid(True)

    # Show the plot
    plt.show()


def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def merge_coordinates(data, radius):
    clusters = []
    visited = [False] * len(data)

    # Create clusters of close coordinates
    for i in range(len(data)):
        if not visited[i]:
            cluster = []
            stack = [i]
            visited[i] = True

            while stack:
                current = stack.pop()
                cluster.append(current)

                for j in range(len(data)):
                    if not visited[j] and euclidean_distance(data[current], data[j]) <= radius:
                        stack.append(j)
                        visited[j] = True

            clusters.append(cluster)

    # Merge data in each cluster and store in dictionary
    result_dict = {}
    inverted_index = {}
    coordinates = []
    for cluster in clusters:
        x_coords = [data[idx, 0] for idx in cluster]
        y_coords = [data[idx, 1] for idx in cluster]
        merged_values = []

        median_x = round(np.median(x_coords))
        median_y = round(np.median(y_coords))
        median_coords = (median_x, median_y)  # Use tuple for dictionary key

        for idx in cluster:
            values_list = data[idx][1].tolist()  # Assuming tensor has a method tolist()
            merged_values.extend(values_list)
            # Populate the inverted index
            for value in values_list:
                if value in inverted_index:
                    inverted_index[value].add(median_coords)
                else:
                    inverted_index[value] = {median_coords}

        result_dict[median_coords] = list(set(merged_values))
        coordinates.append([median_x, median_y])

    # Convert sets in inverted index to lists for consistency
    for key in inverted_index:
        inverted_index[key] = list(inverted_index[key])

    return coordinates, result_dict, inverted_index


def segment(img, numsuperpixels, compactness, doRGBtoLAB):
    # --------------------------------------------------------------
    # Read image and change image shape from (h,w,c) to (c,h,w)
    # --------------------------------------------------------------

    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    dims = img.shape
    h, w, c = dims[0], dims[1], dims[2]
    img = img.reshape(-1).astype(np.double) # Convert the image into a 1D array
    boundaries = np.zeros(h * w, dtype=np.int32)
    labels = np.zeros(h * w, dtype=np.int32)
    numlabels = np.zeros(1, dtype=np.int32)


    # --------------------------------------------------------------
    # Prepare the pointers to pass to the C function
    # --------------------------------------------------------------
    ffibuilder = FFI()
    pinp = ffibuilder.cast("double*", ffibuilder.from_buffer(img))
    plabels = ffibuilder.cast("int*", ffibuilder.from_buffer(labels))
    pnumlabels = ffibuilder.cast("int*", ffibuilder.from_buffer(numlabels))
    pboundaries = ffibuilder.cast("int*", ffibuilder.from_buffer(boundaries.reshape(-1)))

    # Allocate memory for centroids and colors
    kx = np.zeros(numsuperpixels, dtype=np.double)  # centroid x
    ky = np.zeros(numsuperpixels, dtype=np.double)  # centroid y
    ksize = np.zeros(numsuperpixels, dtype=np.double)
    kc = np.zeros(numsuperpixels * c, dtype=np.double)  # centroid color
    kstd = np.zeros(numsuperpixels, dtype=np.double)  # centroid color std 
    kcov = np.zeros(numsuperpixels, dtype=np.double)  # centroid color covariance

    pkx = ffibuilder.cast("double*", ffibuilder.from_buffer(kx))
    pky = ffibuilder.cast("double*", ffibuilder.from_buffer(ky))
    pksize = ffibuilder.cast("double*", ffibuilder.from_buffer(ksize))
    pkc = ffibuilder.cast("double*", ffibuilder.from_buffer(kc))
    pkstd = ffibuilder.cast("double*", ffibuilder.from_buffer(kstd))
    pkcov = ffibuilder.cast("double*", ffibuilder.from_buffer(kcov))
    # --------------------------------------------------------------
    # Call the C function
    # --------------------------------------------------------------
    start = timer()
    SNIC_main(pinp, w, h, c, numsuperpixels, 
              compactness, doRGBtoLAB, plabels, pnumlabels, 
              pkx, pky, pksize, pkc, pboundaries, pkstd, pkcov)
    end = timer()
    # print("number of superpixels: ", numlabels[0])
    # print("time taken in seconds: ", end - start)
    
    centroids = np.column_stack((ky, kx))
    
    # plt.scatter(np.array(centroids).astype(np.int64)[:,1],np.array(centroids).astype(np.int64)[:,0], s=0.1)
    # plt.imshow(labels.reshape(h, w))
    # plt.show()

    # imglabel = np.zeros((h * w))
    # tkc = kc.reshape(kc.shape[0]//c ,c)
    # for i in range(tkc.shape[0]):
    #     imglabel = np.where(labels == i, tkc[i,0], imglabel)
    # plt.scatter(np.array(centroids).astype(np.int64)[:,1],np.array(centroids).astype(np.int64)[:,0], s=0.1)
    # plt.imshow(imglabel.reshape(h, w))
    # plt.show()

    return labels.reshape(h, w), numlabels[0], centroids, kc, kstd, kcov, boundaries


def voronoi_tessellation_fn(image, k=1024, compactness=0.5, doRGBtoLAB=False):

    if 2 == len(image.shape):
        # we will consider the image is a grayscale. So we will add a third dimension on it
        image = torch.unsqueeze(image, dim=-1)
        # check for the dimension of the image which channels have been appeared on the first axis or ont he third axis
    if image.shape[0] <= 3:
        # change the axis of the color intensity to the latest dimension
        image = torch.permute(image, (1, 2, 0))  # FOR COLORFUL IMAGES

    h = image.shape[0]
    w = image.shape[1]

    startSNICTime = time.time()
    label_matrix, num_superpixels, snic_centroids, cents_colors, cents_std, cents_cov, boundaries = segment(img=image, numsuperpixels=k,
                                                                      compactness=compactness, doRGBtoLAB=doRGBtoLAB)
    endSNICTime = time.time()
    # print(f"Time taken for calculating the snic is {endSNICTime - startSNICTime}")
    label_matrix = torch.tensor(label_matrix)
    
    starttime = time.time()
    # lblToCentroids = [] # list of [x, y, color, label]
    dict_lbl_cents = {} # dictionary of {label: [x, y, color]}
    snic_centroids_int = np.round(snic_centroids, decimals=0).astype(int)
    for i in range (num_superpixels):
        dict_lbl_cents[
            i, label_matrix[snic_centroids_int[i,0], snic_centroids_int[i,1]].item()] = \
                [snic_centroids_int[i, 0], snic_centroids_int[i, 1], 
                 cents_colors[i].item(), cents_colors[num_superpixels + i].item(), cents_colors[2 * num_superpixels + i].item(), 
                 cents_std[i].item(), cents_cov[i].item()]
        
    # test the dictionary
    # for key, value in dict_lbl_cents.items():
    #     print(key, value)
    
    # creating the delaunay triangulation EDGES
    edge_labels = set() # set of [label1, label2]
    boundaries_indices = np.where(boundaries==1)[0]
    for i, bndrIdx in enumerate(boundaries_indices):
        x = bndrIdx % w
        y = bndrIdx // w
        
        if x >= 1 and x <= w-2 and y >= 1 and y <= h-2:
            edge_labels.add((
                label_matrix[y.item(), x.item() - 1].item(),
                label_matrix[y.item(), x.item() + 1].item()
            ))
            edge_labels.add((
                label_matrix[y.item() - 1, x.item()].item(),
                label_matrix[y.item() + 1, x.item()].item()
            ))
            edge_labels.add((
                label_matrix[y.item() - 1, x.item() - 1].item(),
                label_matrix[y.item() + 1, x.item() + 1].item()
            ))
            edge_labels.add((
                label_matrix[y.item() - 1, x.item() + 1].item(),
                label_matrix[y.item() + 1, x.item() - 1].item()
            ))

    
    # creating the delaunay triangulation NODES
    # We will use the dict_lbl_cents and the edge_labels to create the nodes and the edges of the delaunay triangulation
    # In the following for loop We will create a graph with the nodes and the edges using the networkx library and the Data object from the torch_geometric library
    DT_Data = Data()
    DT_Data.x = torch.zeros(len(dict_lbl_cents), 7)
    DT_Data.pos = torch.zeros(len(dict_lbl_cents), 2)
    for i, node_idx in enumerate(dict_lbl_cents):
        try:
            DT_Data.x[node_idx[1]] = torch.tensor(dict_lbl_cents[node_idx])
            DT_Data.pos[node_idx[1]] = torch.tensor(dict_lbl_cents[node_idx][:2])
        except Exception as e:
            ...   # some nodes will be missed because of the boundary pixels

    DT_Data.edge_index = torch.tensor(list(edge_labels)).T
    
    
    endtime = time.time()
    # print(f"Time taken for creating the delaunay triangulation is {endtime - starttime}")
    return DT_Data, label_matrix



# print(os.getcwd())
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

# if os.path.isdir('dataset') == False:
#     os.mkdir('dataset')

# NUM_GP = 1024
# # image_name = 'D1.png'
# image_name = '1.jpg'

# # Reading the image using OpenCV and converting it to LAB color space including the normalization
# # image = cv2.imread(f'dataset/cores_BC07001/{image_name}')
# image = cv2.imread(f'{image_name}')
# image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  #.astype(np.float32)  # convert to LAB color space from absolutely BGR uint8 which is [0..255]
# # min_max normalization
# image_lab = imageMinMaxNormalization(image_lab)
# time_ = time.time()

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ main part of the code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Delaunay_tri_graph, label_matrix = voronoi_tessellation_fn(image=image_lab, k=NUM_GP, compactness=1.0, doRGBtoLAB=False)
# Delaunay_tri_graph.edge_index = remove_self_loops(Delaunay_tri_graph.edge_index)[0]
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# print(Delaunay_tri_graph)
# print('The fucking time is : ',time.time() - time_)

# # Show boundaries on the image
# plt.imshow(drawBoundaries(f'dataset/cores_BC07001/{image_name}', label_matrix))

# # plto the delaunay triangulation
# plot_pyg_data_with_pos(
#     data=Delaunay_tri_graph, image=image_lab, node_size=0.1,
#     show_fig=True, save_fig=False, fig_size=(8, 8),
#     fig_text=f'Image: {image_name}__num_V__{Delaunay_tri_graph.x.shape[0]}',
#     fig_name=f'plots/{image_name}__num_V__{Delaunay_tri_graph.x.shape[0]}__{datetime.now()}.png')


# print('The end!...')
