import jpype

import jpype.imports
from jpype.types import *

from _snic.lib import SNIC_main

import numpy as np

# start JVM
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=["/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/model_1_DTGCN_Class/ProcessingUsingJavaCode/src"])


# Import Java classes
from delaunayTriangulationJava import DelaunayGraphBuilder, SuperpixelResult

# Loading an image
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("/Users/mustafamohammadi/Documents/GitHub/3DVNN_202400801/BIOMEDIC_PROJECT/model_1_DTGCN_Class/ProcessingUsingJavaCode/1.jpg")
plt.imshow(img)
print(img.shape)
dims = img.shape
print(dims[0], dims[1], dims[2])


# Flatten image for C interface


# Prepare array for C function output
from cffi import FFI
from timeit import default_timer as timer


numsuperpixels = 1024
compactness = 0.5
doRGBtoLAB = True


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
print("time taken in seconds: ", end - start)

centroids = np.column_stack((ky, kx))


# Convert Python arrays to Java arrays
JLabels = JArray(JInt)(labels.tolist())
JBoundaries = JArray(JInt)(boundaries.tolist())
JCentroids = JArray(JArray(JDouble))([JArray(JDouble)(row) for row in centroids])
JKc = JArray(JDouble)(kc.tolist())
JStd = JArray(JDouble)(kstd.tolist())
JCov = JArray(JDouble)(kcov.tolist())

# Create Java object to pass to Delaunay builder
JavaResult = SuperpixelResult(
    JLabels,
    int(numlabels[0]),
    JCentroids,
    JKc,
    JStd,
    JCov,
    JBoundaries,
    h,
    w
)
print("Done")

# -------------------------------------------------------
# Call Delaunay triangulation from Java
# -------------------------------------------------------
GraphResult = DelaunayGraphBuilder.buildGraph(JavaResult)

# Convert returned values to NumPy
x = np.array([[v for v in row] for row in GraphResult.x])
pos = np.array([[v for v in row] for row in GraphResult.pos])
edge_index = np.array([[v for v in row] for row in GraphResult.edgeIndex])
label_matrix = np.array([[v for v in row] for row in GraphResult.labelMatrix])

# -------------------------------------------------------
# Done! Print shape summary
# -------------------------------------------------------
print(f"x shape: {x.shape}")
print(f"pos shape: {pos.shape}")
print(f"edge_index shape: {edge_index.shape}")
print(f"label_matrix shape: {label_matrix.shape}")
end = timer()
# print("number of superpixels: ", numlabels[0])
print("time taken in seconds: ", end - start)


# Optionally shut down JVM
jpype.shutdownJVM()