package delaunayTriangulationJava;

public class GraphResult {
    public double[][] x;             // Node features
    public int[][] edgeIndex;        // Edges
    public double[][] pos;           // Node coordinates
    public int[][] labelMatrix;      // Segmentation labels

    public GraphResult(double[][] x, int[][] edgeIndex, double[][] pos, int[][] labelMatrix) {
        this.x = x;
        this.edgeIndex = edgeIndex;
        this.pos = pos;
        this.labelMatrix = labelMatrix;
    }
}
