// #ifndef SNIC_H
// #define SNIC_H

// #ifdef __cplusplus
// extern "C" {
// #endif

/*
 * Structures for SNIC's internal heap and nodes
 */
typedef struct {
    unsigned int i;  // pixel index (x and y packed)
    unsigned int k;  // superpixel label
    double       d;  // distance (priority)
} NODE;

typedef struct {
    NODE* nodes; // array of NODE
    int   len;   // current length of the heap
    int   size;  // current allocated size of the heap
} HEAP;

/*
 * Functions for maintaining the min-heap used in SNIC
 */
void push(HEAP *h, const unsigned int ind, const unsigned int klab, const double dist);
void pop (HEAP* h, unsigned int* ind, unsigned int* klab, double* dist);

/*
 * Convert RGB arrays [0..255] to LAB
 * Only used if there are exactly 3 channels (R, G, B).
 */
void rgbtolab(double* rin, double* gin, double* bin, double sz,
              double* lvec, double* avec, double* bvec);

/*
 * FindSeeds
 * Computes approximate grid steps and picks seed centers for the superpixels.
 */
void FindSeeds(const int width, const int height, const int numk,
               int* kx, int* ky, int* outnumk);

/*
 * runSNIC
 * Performs the SNIC-based superpixel segmentation.
 * 
 * Inputs:
 *   chans      - double** to each channel of the image data
 *   nchans     - number of channels
 *   width, height
 *   labels     - output labels array (size: width*height)
 *   outnumk    - actual number of superpixels discovered
 *   innumk     - initial (requested) superpixel count
 *   compactness
 *
 * Outputs:
 *   out_kx, out_ky      - superpixel centroids (x and y)
 *   out_ksize           - number of pixels in each superpixel
 *   out_kc              - mean color per channel for each superpixel
 *   out_std, out_cov    - pooled std and pooled covariance for each superpixel
 *                         (only valid if nchans == 3, otherwise set to 0)
 */
void runSNIC(double** chans, const int nchans,
             const int width, const int height,
             int* labels, int* outnumk, const int innumk, const double compactness,
             double** out_kx, double** out_ky, double** out_ksize,
             double*** out_kc,
             double* out_std,  // pooled std (one value per superpixel)
             double* out_cov   // pooled covariance (one value per superpixel)
             );

/*
 * boundary_detection
 * Given the final labels, marks pixels that are on superpixel boundaries.
 * boundaries_out[i] = 1 if pixel i is on boundary, else 0.
 */
void boundary_detection(int* labels, const int width, const int height,
                        int* boundaries_out);

/*
 * Delaunay_triangulation
 * Example function for building a Delaunay-like graph of superpixel centroids.
 * Not modified by the standard deviation / covariance additions.
 */
void Delaunay_triangulation(const int intWidth, const int intHeight, 
                            int* pArrLabels, double* pArrKX, double* pArrKY,
                            int intNumK, 
                            int* pArrBoundaries,
                            int* pArrLblToCentroidsOut,
                            int* pArrEdgeLblsOut);

/*
 * SNIC_main
 * High-level function that:
 *   1) Optionally converts RGB->LAB (if doRGBtoLAB && nchannels == 3),
 *   2) Runs SNIC to get superpixel labels,
 *   3) Computes boundary mask,
 *   4) Copies outputs to user arrays,
 *   5) Frees intermediate memory.
 *
 * Inputs/Outputs:
 *   img          - pointer to raw image data (double),
 *                  typically sized [nchannels * width * height]
 *   width, height
 *   nchannels    - number of channels
 *   numSuperpixels
 *   compactness
 *   doRGBtoLAB   - 1 to convert before SNIC, 0 otherwise
 *   klabels      - pointer to user-allocated array [width*height] for final labels
 *   numlabels    - pointer that will hold the actual number of superpixels used
 *   kx_out, ky_out, ksize_out - arrays to store centroid coords, sizes
 *   kc_out        - array for storing flattened color means 
 *                   (dimension: [numSuperpixels * nchannels])
 *   boundaries    - array [width*height] for storing boundary mask
 *   kstd_out      - array [numSuperpixels] for pooled std
 *   kcov_out      - array [numSuperpixels] for pooled cov
 */
void SNIC_main(double* img, 
               const int width, 
               const int height,
               const int nchannels, 
               const int numSuperpixels, 
               const double compactness,
               const int doRGBtoLAB,
               int* klabels,
               int* numlabels,
               double* kx_out,
               double* ky_out,
               double* ksize_out,
               double* kc_out,
               int* boundaries,
               double* kstd_out,
               double* kcov_out
               );

// #ifdef __cplusplus
// }
// #endif

// #endif  // SNIC_H
