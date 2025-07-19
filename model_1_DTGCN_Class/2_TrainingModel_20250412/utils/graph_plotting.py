import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


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
