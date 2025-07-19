package delaunayTriangulationJava;

//import java.util.Arrays;

public class SuperpixelSegmenter {
    static {
        System.loadLibrary("snic_native");  // Load your native library
    }

    public native void snic(double[] img, int w, int h, int c, int numSuperpixels, double compactness, boolean lab, 
                            int[] labels, int[] numLabels, double[] kx, double[] ky, double[] ksize, 
                            double[] kc, int[] boundaries, double[] kstd, double[] kcov);

    public static SuperpixelResult segment(double[][][] img, int numSuperpixels, double compactness, boolean lab) {
        int h = img.length, w = img[0].length, c = img[0][0].length;
        double[] flatImg = new double[h * w * c];
        int idx = 0;
        for (double[][] row : img)
            for (double[] px : row)
                for (double v : px)
                    flatImg[idx++] = v;

        int[] labels = new int[h * w];
        int[] boundaries = new int[h * w];
        int[] numLabels = new int[1];
        double[] kx = new double[numSuperpixels];
        double[] ky = new double[numSuperpixels];
        double[] ksize = new double[numSuperpixels];
        double[] kc = new double[numSuperpixels * c];
        double[] kstd = new double[numSuperpixels];
        double[] kcov = new double[numSuperpixels];

        SuperpixelSegmenter s = new SuperpixelSegmenter();
        s.snic(flatImg, w, h, c, numSuperpixels, compactness, lab, labels, numLabels, kx, ky, ksize, kc, boundaries, kstd, kcov);

        double[][] centroids = new double[numSuperpixels][2];
        for (int i = 0; i < numSuperpixels; i++) {
            centroids[i][0] = ky[i];
            centroids[i][1] = kx[i];
        }

        return new SuperpixelResult(labels, numLabels[0], centroids, kc, kstd, kcov, boundaries, h, w);
    }
}

