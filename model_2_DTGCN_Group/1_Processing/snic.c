#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "snic.h"

/*
HEAP DATA STRUCTURE IMPLEMENTATION
*/

void push (HEAP *h, const unsigned int ind, const unsigned int klab, const double dist)
{
    if (h->len + 1 >= h->size)
    {
        h->size = h->size ? h->size * 2 : 4;
        h->nodes = (NODE *)realloc(h->nodes, h->size * sizeof (NODE));
    }
    int i = h->len + 1;
    int j = i / 2;
    while (i > 1 && h->nodes[j].d > dist)
    {
        h->nodes[i] = h->nodes[j];
        i = j;
        j = j / 2;
    }
    h->nodes[i].i = ind;
    h->nodes[i].k = klab;
    h->nodes[i].d = dist;
    h->len++;
}
 
void pop(HEAP* h, unsigned int* ind, unsigned int* klab, double* dist)
{
    if(h->len > 1)
    {
        int i, j, k;
        *ind  = h->nodes[1].i;
        *klab = h->nodes[1].k;
        *dist = h->nodes[1].d;
     
        h->nodes[1] = h->nodes[h->len];
        h->len--;
     
        i = 1;
        while (i != h->len+1)
        {
            k = h->len+1;
            j = 2 * i;
            if (j <= h->len && h->nodes[j].d < h->nodes[k].d)
            {
                k = j;
            }
            if (j + 1 <= h->len && h->nodes[j + 1].d < h->nodes[k].d)
            {
                k = j + 1;
            }
            h->nodes[i] = h->nodes[k];
            i = k;
        }
    }
}


/*
RGB TO LAB CONVERSION
*/
void rgbtolab(double* rin, double* gin, double* bin, double sz, double* lvec, double* avec, double* bvec)
{
    int i;
    double sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856; 
    const double kappa   = 903.3;	  
    
    const double Xr = 0.950456;	 
    const double Yr = 1.0;		   
    const double Zr = 1.088754;	
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)	r = R/12.92;
        else				r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)	g = G/12.92;
        else				g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)	b = B/12.92;
        else				b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        
        //------------------------
        // XYZ to LAB conversion
        //------------------------
        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
        else				fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
        else				fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
        else				fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; 
        avec[i] = aval; 
        bvec[i] = bval;
    }
}

/*
FindSeeds
Computes approximate grid steps and picks seed centers for the superpixels.
*/
void FindSeeds(const int width, const int height, const int numk, int* kx, int* ky, int* outnumk)
{
    const int sz = width*height;
    double gridstep = sqrt((double)(sz)/(double)(numk)) + 0.5;

    // Try to adjust the step slightly to get closer to the desired numk
    {
        int minerr = 999999;
        double minstep = gridstep-1.0; 
        double maxstep = gridstep+1.0;
        for(double x = minstep; x <= maxstep; x += 0.1)
        {
            int err = abs( (int)(0.5 + width/x)*(int)(0.5 + height/x) - numk);
            if(err < minerr)
            {
                minerr = err; 
                gridstep = x;
            }
        }
    }

    double halfstep = gridstep/2.0;
    int n = 0;
    for(double y = halfstep; y <= height; y += gridstep)
    {
        int yval = (int)(y+0.5);
        if(yval < height)
        {
            for(double x = halfstep; x <= width; x += gridstep)
            {
                int xval = (int)(x+0.5);
                if(xval < width)
                {
                    kx[n] = xval;
                    ky[n] = yval;
                    n++;
                }
            }
        }
    }
    *outnumk = n;
}

/* 
runSNIC
Performs the SNIC-based superpixel segmentation. 
NEW: This version also computes the pooled standard deviation and covariance 
     for 3-channel images (assumed to be L,a,b).
*/
void runSNIC(
            double**  chans,         // channel pointers
            const int nchans,        // number of channels
            const int width,
            const int height,
            int*      labels,        // [width*height] output: superpixel labels
            int*      outnumk,       // actual number of superpixels used
            const int innumk,        // requested num. of superpixels
            const double compactness,
            double**   out_kx,       // [numk]
            double**   out_ky,       // [numk]
            double**   out_ksize,    // [numk]
            double***  out_kc        // [nchans][numk]
            ,
            double*    out_std,      // [numk]   <-- newly added for standard deviation
            double*    out_cov       // [numk]   <-- newly added for covariance
            )
{
    const int w = width;
    const int h = height;
    const int sz = w*h;
    const int dx8[8] = {-1,  0, 1, 0, -1,  1, 1, -1};
    const int dy8[8] = { 0, -1, 0, 1, -1, -1, 1,  1};
    const int dn8[8] = {-1, -w, 1, w, -1-w,1-w,1+w,-1+w};
    
    // 1) Find initial seeds
    int* cx = (int*)malloc( sizeof(int) * (int)(innumk * 1.1 + 10) );
    int* cy = (int*)malloc( sizeof(int) * (int)(innumk * 1.1 + 10) );
    int numk = 0;
    FindSeeds(w, h, innumk, cx, cy, &numk);

    // 2) Init HEAP for the clustering
    HEAP* heap = (HEAP *)calloc(1, sizeof (HEAP));
    heap->nodes = (NODE*)calloc(sz,sizeof(NODE));
    heap->len = 0; 
    heap->size = sz;

    // 3) Label init
    for(int i = 0; i < sz; i++) {
        labels[i] = -1;
    }

    // 4) Insert seeds into the heap
    for(int k = 0; k < numk; k++)
    {
        push(heap,(cx[k] << 16 | cy[k]),k,0.0);
    }
    
    // 5) Allocate memory for the centroid accumulators
    *out_kx    = (double*)calloc(numk,sizeof(double));
    *out_ky    = (double*)calloc(numk,sizeof(double));
    *out_ksize = (double*)calloc(numk,sizeof(double));
    
    *out_kc = (double**)malloc(sizeof(double*)*nchans);
    for(int c = 0; c < nchans; c++)
    {
        (*out_kc)[c] = (double*)calloc(numk,sizeof(double));
    }

    // 6) ADDITIONAL ARRAYS for standard deviation / covariance computations
    //    Only do this if the image has 3 channels (L,a,b)
    //    sumSq: sum of squares for each channel
    //    sumCrossXY: sum of cross-terms for pairs of channels
    double *sumL2=NULL, *sumA2=NULL, *sumB2=NULL;
    double *sumLA=NULL, *sumLB=NULL, *sumAB=NULL;
    if(nchans == 3)
    {
        sumL2 = (double*)calloc(numk,sizeof(double));
        sumA2 = (double*)calloc(numk,sizeof(double));
        sumB2 = (double*)calloc(numk,sizeof(double));
        sumLA = (double*)calloc(numk,sizeof(double));
        sumLB = (double*)calloc(numk,sizeof(double));
        sumAB = (double*)calloc(numk,sizeof(double));
    }

    // 7) Main SNIC loop
    int pixelcount = 0;
    const int CONNECTIVITY = 4;
    const double M = compactness;
    const double invwt = (M*M*numk)/(double)(sz);
    
    unsigned int ind;
    unsigned int klab;
    double dist;
    int loopcount = 0;
    
    while(pixelcount < sz)
    {
        pop(heap, &ind, &klab, &dist);
        int k = (int) klab;
        int x = (ind >> 16) & 0xffff;
        int y = ind & 0xffff;
        int i = y*width + x;
        
        if(labels[i] < 0)
        {
            // Assign pixel i to superpixel k
            labels[i] = k; 
            pixelcount++;

            // Accumulate color values
            for(int c = 0; c < nchans; c++)
            {
                (*out_kc)[c][k] += chans[c][i];
            }
            // Accumulate x,y
            (*out_kx)[k] += x;
            (*out_ky)[k] += y;
            (*out_ksize)[k] += 1.0;

            // If 3 channels (L,a,b), accumulate squares & cross-terms:
            if(nchans == 3)
            {
                double Lval = chans[0][i];
                double Aval = chans[1][i];
                double Bval = chans[2][i];

                sumL2[k] += (Lval*Lval);
                sumA2[k] += (Aval*Aval);
                sumB2[k] += (Bval*Bval);

                sumLA[k] += (Lval*Aval);
                sumLB[k] += (Lval*Bval);
                sumAB[k] += (Aval*Bval);
            }

            // Explore neighbors
            for(int p = 0; p < CONNECTIVITY; p++)
            {
                int xx = x + dx8[p];
                int yy = y + dy8[p];
                if(!(xx < 0 || xx >= w || yy < 0 || yy >= h))
                {
                    int ii = i + dn8[p];
                    if(labels[ii] < 0)
                    {
                        double colordist = 0.0;
                        for(int c = 0; c < nchans; c++)
                        {
                            double cdiff = (*out_kc)[c][k] - (chans[c][ii] * (*out_ksize)[k]);
                            colordist += (cdiff*cdiff);
                        }
                        double xdiff = (*out_kx)[k] - xx*(*out_ksize)[k];
                        double ydiff = (*out_ky)[k] - yy*(*out_ksize)[k];
                        double xydist = xdiff*xdiff + ydiff*ydiff;

                        double slicdist = (colordist + xydist*invwt) / 
                                          ((*out_ksize)[k]*(*out_ksize)[k]);

                        push(heap, (xx << 16 | yy), k, slicdist);
                    }
                }
            }
        }
        loopcount++;
        if(loopcount > sz*10)
        {
            printf("Error: SNIC loop count exceeded a large threshold.\n");
            break;
        }
    }

    *outnumk = numk;

    // 8) Final normalization of means
    for (int k = 0; k < numk; k++) 
    {
        double count = (*out_ksize)[k];
        if (count > 0.0) 
        {
            (*out_kx)[k] /= count;
            (*out_ky)[k] /= count;
            for (int c = 0; c < nchans; c++) 
            {
                (*out_kc)[c][k] /= count;
            }
        }
    }

    // 9) Compute the STD and COV if we have 3 channels.
    //    We'll define the "pooled" standard deviation as the sqrt of the average
    //    of Var(L), Var(a), Var(b). We'll define the "pooled" covariance as the average
    //    of Cov(L,a), Cov(L,b), Cov(a,b).
    if(nchans == 3)
    {
        for(int k=0; k < numk; k++)
        {
            double count = (*out_ksize)[k];
            if(count > 1.0)
            {
                // Means
                double mL = (*out_kc)[0][k];
                double mA = (*out_kc)[1][k];
                double mB = (*out_kc)[2][k];
                
                // E[L^2] = sumL2[k]/count, etc.
                double eL2 = sumL2[k]/count;
                double eA2 = sumA2[k]/count;
                double eB2 = sumB2[k]/count;

                double eLA = sumLA[k]/count; 
                double eLB = sumLB[k]/count; 
                double eAB = sumAB[k]/count;

                // Variances
                double varL = eL2 - (mL*mL);
                double varA = eA2 - (mA*mA);
                double varB = eB2 - (mB*mB);

                // Covariances
                double covLA = eLA - (mL*mA);
                double covLB = eLB - (mL*mB);
                double covAB = eAB - (mA*mB);

                // "Pooled" standard deviation across L,a,b
                double avgVar = (varL + varA + varB)/3.0;
                if(avgVar < 0.0) avgVar = 0.0; // numeric safety
                out_std[k] = sqrt(avgVar);

                // "Pooled" covariance across (L,a,b)
                out_cov[k] = (covLA + covLB + covAB)/3.0;
            }
            else
            {
                // If a superpixel has 1 or 0 pixels, set them to 0
                out_std[k] = 0.0;
                out_cov[k] = 0.0;
            }
        }
    }
    else
    {
        // For non-3-channel images, we just set these to 0.
        for(int k=0; k<numk; k++)
        {
            out_std[k] = 0.0;
            out_cov[k] = 0.0;
        }
    }

    // Cleanup
    free(cx);
    free(cy);
    if(heap->nodes) free(heap->nodes);
    if(heap) free(heap);

    // free sums for extra statistics
    if(nchans == 3)
    {
        free(sumL2); free(sumA2); free(sumB2);
        free(sumLA); free(sumLB); free(sumAB);
    }
}

/*
boundary_detection
Mark boundary pixels in boundaries_out[] if they differ from a neighbor's label.
*/
void boundary_detection(int* labels, const int width, const int height, int* boundaries_out)
{
    const int sz = width * height;
    const int dx8[4] = {-1,  0, 1, 0}; // 4-neighbors
    const int dy8[4] = { 0, -1, 0, 1};

    for (int i = 0; i < sz; i++) 
    {
        boundaries_out[i] = 0; // init
        int x = i % width;
        int y = i / width;
        int myLabel = labels[i];
        for (int p = 0; p < 4; p++) 
        {
            int xx = x + dx8[p];
            int yy = y + dy8[p];
            if (xx >= 0 && xx < width && yy >= 0 && yy < height) 
            {
                if (labels[yy * width + xx] != myLabel)
                {
                    boundaries_out[i] = 1;
                    break;
                }
            }
        }
    }
}

/*
Delaunay_triangulation
Creates a Delaunay-like graph structure from the superpixel labels and boundaries.

NOTE: This function is not modified for the std/cov extension; 
      it remains as you originally had it.
*/
void Delaunay_triangulation(const int intWidth, const int intHeight, 
    int *pArrLabels, double *pArrKX, double *pArrKY, int intNumK, 
    int *pArrBoundaries, int* pArrLblToCentroidsOut, int *pArrEdgeLblsOut)
{
    int sz = intWidth * intHeight;
    int dx8[8] = {-1,  0, 1, 0, -1,  1, 1, -1};
    int dy8[8] = { 0, -1, 0, 1, -1, -1, 1,  1};

    printf("----------- 1 --------------- \n");
    printf("First loop; Mapping the Centroids to the labels \n");
    printf("width: %d, height: %d, numk: %d \n", intWidth, intHeight, intNumK);
    for (int i = 0; i < intNumK; i++)
    {
        double temp = pArrKY[i] * intWidth + pArrKX[i];
        if (i < intNumK)
        {
            printf("Centroid: %d, y: %f, x: %f --> pArrKY[i](%f) * intWidth(%d) + pArrKX[i](%f) = %f\n", 
                   i, pArrKY[i], pArrKX[i], pArrKY[i], intWidth, pArrKX[i], temp);
            printf("pArrLabels[temp/label]: %d \n", pArrLabels[(int) temp]);
        }
        int label_number = pArrLabels[(int) temp];
        pArrLblToCentroidsOut[i]             = label_number;
        pArrLblToCentroidsOut[intNumK + i]   = (int)pArrKX[i];
        pArrLblToCentroidsOut[2*intNumK + i] = (int)pArrKY[i];
    }
    // testing the output
    for (int i=0; i<intNumK; i++)
    {
        printf("label: %d, x: %d, y: %d \n",
               pArrLblToCentroidsOut[i],
               pArrLblToCentroidsOut[intNumK + i],
               pArrLblToCentroidsOut[2 * intNumK + i]);
    }

    int edges_count_out = 0;
    printf("Running the second for loop \n");
    for (int i = 0; i < sz; i++)
    {
        if (pArrBoundaries[i] != 0)
        {
            int x = i % intWidth;
            int y = i / intWidth;
            for (int p = 0; p < 8; p++)
            {
                int xx = x + dx8[p];
                int yy = y + dy8[p];
                if (xx >= 0 && xx < intWidth && yy >= 0 && yy < intHeight)
                {
                    int reg1 = pArrLabels[i];
                    int reg2 = pArrLabels[yy * intWidth + xx];
                    if (reg1 != reg2)
                    {
                        pArrEdgeLblsOut[edges_count_out] = reg1;
                        pArrEdgeLblsOut[8*intNumK + edges_count_out] = reg2;
                        edges_count_out++;
                    }
                }
            }
        }
    }
    printf("end of the function \n");
}


/*
SNIC_main
High-level function that:
 1) optionally converts RGB -> LAB,
 2) runs SNIC to get superpixel labels,
 3) computes the superpixel boundary mask,
 4) copies outputs to user arrays, 
 5) (optionally) could run Delaunay_triangulation or other steps.

NEW: We also pass out arrays for standard deviation & covariance (each of size numSuperpixels).
*/
void SNIC_main(double* img, 
               const int width, 
               const int height,
               const int nchannels, 
               const int numSuperpixels, 
               const double compactness,
               const int doRGBtoLAB, 
               int* klabels,    // [width*height]
               int* numlabels,  
               double* kx_out,  // [numSuperpixels] for centroid X
               double* ky_out,  // [numSuperpixels] for centroid Y
               double* ksize_out, // [numSuperpixels] for superpixel sizes
               double* kc_out,  // [numSuperpixels * nchannels] flatten
               int* boundaries, // [width*height] boundary mask
               double* kstd_out, // [numSuperpixels] <-- standard deviation
               double* kcov_out  // [numSuperpixels] <-- covariance
               )
{
    // printf("Starting SNIC_main\n");
    int sz = width * height;

    // 1) Prepare channel pointers
    double** channels = (double**)malloc(sizeof(double*) * nchannels);
    for (int c = 0; c < nchannels; c++) {
        channels[c] = img + c * sz;
    }

    // 2) Convert to LAB if asked and if exactly 3 channels
    if (doRGBtoLAB && nchannels == 3) {
        rgbtolab(channels[0], channels[1], channels[2], sz, 
                 channels[0], channels[1], channels[2]);
    }

    // 3) SNIC arrays
    double* kx    = NULL;
    double* ky    = NULL;
    double* ksize = NULL;
    double** kc   = NULL;

    // 4) Run SNIC
    // printf("Running SNIC\n");
    int numklabels = 0;
    runSNIC(channels, nchannels, width, height,
            klabels, &numklabels, numSuperpixels, compactness,
            &kx, &ky, &ksize, &kc,
            kstd_out,  // newly added
            kcov_out); // newly added

    *numlabels = numklabels;

    for (int i = 0; i < numklabels; i++) {
        kx_out[i]    = kx[i];
        ky_out[i]    = ky[i];
        ksize_out[i] = ksize[i];
    }
    
    // 5) Flatten centroid color outputs for the user
    for (int c = 0; c < nchannels; c++) {
        for (int i = 0; i < numklabels; i++) {
            kc_out[c * numklabels + i] = kc[c][i];
        }
    }
    

    // 6) Compute boundary mask
    // printf("Running boundary detection\n");
    boundary_detection(klabels, width, height, boundaries);

    // 7) Free allocated memory from runSNIC outputs
    // printf("Freeing memory\n");
    free(channels);
    free(kx);
    free(ky);
    free(ksize);
    for (int c = 0; c < nchannels; c++) 
    {
        free(kc[c]);
    }
    free(kc);

    // (Optional) You could call Delaunay_triangulation here if you wish, 
    // but it is commented out for clarity.
    /*
    // Example usage:
    // int *lbl_to_centroids = (int*)malloc(sizeof(int) * numklabels * 3);
    // int *edges_label      = (int*)malloc(sizeof(int) * numklabels * 8 * 2);
    // Delaunay_triangulation(width, height, klabels, kx_out, ky_out, 
    //                        numklabels, boundaries, 
    //                        lbl_to_centroids, edges_label);
    // free(lbl_to_centroids);
    // free(edges_label);
    */
}

