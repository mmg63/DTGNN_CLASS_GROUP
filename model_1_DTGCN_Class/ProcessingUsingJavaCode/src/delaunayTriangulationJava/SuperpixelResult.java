package delaunayTriangulationJava;

public class SuperpixelResult {
    public int[] labels;
    public int numLabels;
    public double[][] centroids;
    public double[] colors;
    public double[] std;
    public double[] cov;
    public int[] boundaries;
    public int height, width;

    public SuperpixelResult(int[] labels, int numLabels, double[][] centroids, double[] colors, double[] std, double[] cov, int[] boundaries, int h, int w) {
        this.labels = labels;
        this.numLabels = numLabels;
        this.centroids = centroids;
        this.colors = colors;
        this.std = std;
        this.cov = cov;
        this.boundaries = boundaries;
        this.height = h;
        this.width = w;
    }
}
