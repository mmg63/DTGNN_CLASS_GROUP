import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from skimage.io import imread,imshow


def find_neighbors(label_map, x, y):
    """Find the unique labels of neighboring pixels around (x, y).
    Args:
        label_map: 2D numpy array of labels.
        x: X-coordinate of the pixel.
        y: Y-coordinate of the pixel.
    Returns:
        Set of unique labels of neighboring pixels.
    """
    neighbors = set()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the center pixel itself
            nx_, ny = x + dx, y + dy
            if 0 <= nx_ < label_map.shape[0] and 0 <= ny < label_map.shape[1]:
                neighbors.add(label_map[ny, nx_])
    return neighbors

def graph_tessellation(voronoi_graph, label_map, centroids, plot_graph=True):
    """
    Generate a tessellation graph from a Voronoi graph.
    Args:
        voronoi_graph: Voronoi graph.
        label_map: Label map.
        centroids: Centroids of the regions.
        plot_graph: Whether to plot the tessellation graph.
    Returns:
        Tessellation graph.
    """    
    # Calculating the Dictionary of Voronoi vertices and regions
    dict_vor_vert_regions = dict()
    for x, y in voronoi_graph[0][0].keys():
        neighbors = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the center pixel itself
                nx_, ny = x + dx, y + dy
                try:#if 0 <= nx_ < label_map.shape[0] and 0 <= ny < label_map.shape[1]:
                    neighbors.add(label_map[ny, nx_])
                except:
                    ...
        dict_vor_vert_regions[(x, y)] = set(neighbors)
    # Calculating the Centroids to regions and vs
    dict_cent_region = dict()
    dict_region_cent = dict()
    for cent in centroids:
        x, y = np.round(cent[0])
        x, y = int(x), int(y)
        dict_cent_region[(x, y)] = label_map[y, x]
        dict_region_cent[label_map[y, x]] = cent
    # for x, y in dict_cent_region.keys():
    #     plt.scatter(x, y)
    
    # Calculating edges for the tessellation graph
    tess_graph_geo = list()
    tess_graph_nx = nx.Graph()
    for vor_ver_i, vor_ver_j in voronoi_graph[0][1]:
        # finding the other regions rather that the shared region in an edge in the voronoi diagram
        connect_regions = list(dict_vor_vert_regions[vor_ver_i].intersection(dict_vor_vert_regions[vor_ver_j]))
        # try:
        if len(connect_regions) >= 2:
            region_i, region_j = connect_regions[0], connect_regions[1]
            # except:
            #     ...
            try:
                tess_graph_nx.add_node(region_i,
                                       pos=(dict_region_cent[region_i][0]),
                                       x=torch.asarray(dict_region_cent[region_i][0] + dict_region_cent[region_i][1]))
                tess_graph_nx.add_node(region_j,
                                       pos=(dict_region_cent[region_j][0]),
                                       x=torch.asarray(dict_region_cent[region_j][0] + dict_region_cent[region_j][1]))
                tess_graph_nx.add_edge(region_i, region_j)
            except KeyError:
                continue
    if plot_graph:
        pos = {node: data['pos'] for node, data in tess_graph_nx.nodes(data=True)}
        nx.draw(tess_graph_nx, pos=pos, node_color='lightblue', edge_color='gray')
    # Convert networkx into pytorch geometric graph
    # ***********************************************
    # [ ] Convert networkx into pytorch geometric graph
    # Step 1: Create a mapping from old indices to new, consecutive indices
    node_mapping = {node: i for i, node in enumerate(tess_graph_nx.nodes())}

    # Step 2: Update edge indices based on the new node mapping
    edges = list(tess_graph_nx.edges())
    edge_index = torch.tensor([[node_mapping[i], node_mapping[j]] for i, j in edges], dtype=torch.long).t().contiguous()
    # Edge weights
    edge_attr = []
    try:
        for edge_ in tess_graph_nx.edges(data=True):
            edge_attr.append(edge_[2]['weight'])
            edge_attr = torch.tensor(edge_attr)
    except:
        ...

    # Step 3: Organize node features according to the new indexing
    num_features = len(next(iter(tess_graph_nx.nodes(data=True)))[1]['x'])  # Assuming all nodes have features
    node_features = torch.zeros((len(tess_graph_nx.nodes()), num_features))  # Initialize feature tensor
    for node, data in tess_graph_nx.nodes(data=True):
        node_features[node_mapping[node]] = torch.tensor(data['x'])
    # Step 4: Organize positions
    positions = torch.tensor([tess_graph_nx.nodes[n]['pos'] for n in tess_graph_nx.nodes()], dtype=torch.float)
    # Step 4: Create the PyTorch Geometric Data object
    tess_graph_geo = Data(x=node_features, edge_index=edge_index, pos=positions)  #, edge_attr=edge_attr, )

    if plot_graph:
        plot_pyg_data_with_pos(tess_graph_geo)
    return tess_graph_nx, tess_graph_geo

def plot_pyg_data_with_pos(
        data, 
        image=None, 
        node_color='green', 
        edge_color='blue', 
        node_size=1,
        show_fig=False,
        save_fig=False,
        fig_name='graph.png',
        fig_text = '',
        fig_size=(8, 8),
        fig_res=300
    ):
    """
    Plot a PyTorch Geometric Data object with node positions.
    Args:
        data: PyTorch Geometric Data object.
        image: Image to overlay the graph on.
        node_color: Node color.
        edge_color: Edge color.
        node_size: Node size.
        show_fig: Whether to display the figure.
        save_fig: Whether to save the figure.
        fig_name: Name of the figure to save.
        fig_text: Text to display on the figure.
        fig_size: Size of the figure.
        fig_res: Resolution of the figure.
    """
    # Create a NetworkX graph from the PyTorch Geometric Data object
    G = nx.Graph()
    plt.figure(1, figsize=fig_size)

    edge_index = data.edge_index.numpy()
    edges = zip(edge_index[0], edge_index[1])
    G.add_edges_from(edges)

    pos = None
    # if data.pos is not None:
    if data.pos is not None:
        pos = {i: pos for i, pos in enumerate(data.pos.tolist())}
    else:
        # Generate positions for plotting if not provided
        pos = nx.spring_layout(G)

    edge_labels = {}
    if data.edge_attr is not None:
        edge_labels = {}
        for i, attr in enumerate(data.edge_attr):
            edge_labels[(int(data.edge_index[0, i])), (int(data.edge_index[1, i]))] = attr.item()

    # Plotting
    # plt.figure(figsize=(8, 8))
    if image is not None:
        plt.xticks(range(0, image.shape[0], 1))
        plt.yticks(range(0, image.shape[1], 1))
        plt.imshow(image)
    
    if data.edge_attr is not None:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, with_labels=False)
    else:
        nx.draw(G, pos, with_labels=False, node_color=node_color, edge_color=edge_color, node_size=node_size)
            
    plt.title('Delaunay Triangulation Visualization')
    plt.figtext(0.5, 0.01, fig_text, ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Save the figure if required
    if save_fig:
        plt.savefig(fig_name, dpi=fig_res)

    if show_fig:
        plt.show()
    else:
        plt.close()



def plot_nx_with_pos(graph, image, node_color='red', edge_color='black', node_size=10, with_label=False):
    """
    Plot a NetworkX graph with node positions.
    Args:
        graph: NetworkX graph object.
        image: Image to overlay the graph on.
        node_color: Node color.
        edge_color: Edge color.
        node_size: Node size.
        with_label: Whether to display node labels.
    """
    pos = {node: data['pos'] for node, data in graph.nodes(data=True)}

    if image is not None:
        plt.xticks(range(0, image.shape[0], 1))
        plt.yticks(range(0, image.shape[1], 1))
        plt.imshow(image)
    
    nx.draw(graph, pos, with_labels=with_label, node_color=node_color, edge_color=edge_color, node_size=node_size)
    
    plt.title('NetworkX Graph Visualization')
    plt.show()

def plot_3d(x, y, z):
    """
    Plot a 3D network graph.
    Args:
        x: X-coordinates of the nodes.
        y: Y-coordinates of the nodes.
        z: Z-coordinates of the nodes.
    """
    # Convert to numpy arrays
    x, y, z = np.array(x), np.array(y), np.array(z),
    
    # Create trace
    trace = go.Scatter3d(
        x=np.array(x),
        y=np.array(y),
        z=np.array(z),
        mode='markers+lines',
        marker=dict(
            size=5,
            color=z,  # Color by z-value
            colorscale='Viridis',
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="3D Network Graph",
        width=700,
        height=700,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def plot_boundary_3d(_3DTensorData):
    """
    Plot a 3D surface plot of the input 3D tensor data.
    Args:
        _3DTensorData: 3D tensor data.
    """
    # Get the 3D tensor data
    boundary = _3DTensorData
    
    # Generate x and y values correctly
    x = np.linspace(0, boundary.shape[1] - 1, boundary.shape[1])
    y = np.linspace(0, boundary.shape[0] - 1, boundary.shape[0])
    x, y = np.meshgrid(x, y)
    z = boundary

    # Create 3D plot
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

    # Update layout for better visualization
    fig.update_layout(title='3D Surface Plot', autosize=False,
                    width=700, height=700,
                    margin=dict(l=65, r=50, b=65, t=90))

    # Show the plot
    pio.show(fig)


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

def drawBoundaries(imgname,labels):
    """
    Draw boundaries on the input image based on the input labels.
    Args:
        img: Input image.
        labels: Label map.
        numlabels: Number of unique labels.
    Returns:
        Image with boundaries drawn.
    """
    img = imread(imgname)
    img = np.array(img)
    
    ht, wd = labels.shape

    for y in range(1, ht-1):
        for x in range(1, wd-1):
            if labels[y, x-1] != labels[y, x+1] or labels[y-1, x] != labels[y+1, x]:
                img[y, x, :] = 0

    return img