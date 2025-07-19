package delaunayTriangulationJava;

import java.util.*;
import delaunayTriangulationJava.GraphResult;
import delaunayTriangulationJava.SuperpixelResult;


public class DelaunayGraphBuilder {
    public static GraphResult buildGraph(SuperpixelResult res) {
        int[][] labelMatrix = new int[res.height][res.width];
        for (int i = 0; i < res.height; i++) {
            for (int j = 0; j < res.width; j++) {
                labelMatrix[i][j] = res.labels[i * res.width + j];
            }
        }

        Map<Integer, double[]> nodeFeatures = new HashMap<>();
        Set<String> edgeSet = new HashSet<>();

        for (int i = 0; i < res.numLabels; i++) {
            int y = (int)Math.round(res.centroids[i][0]);
            int x = (int)Math.round(res.centroids[i][1]);
            int label = labelMatrix[y][x];

            nodeFeatures.put(label, new double[]{
                res.centroids[i][0], res.centroids[i][1],
                res.colors[i], res.colors[res.numLabels + i], res.colors[2 * res.numLabels + i],
                res.std[i], res.cov[i]
            });
        }

        for (int i = 0; i < res.height; i++) {
            for (int j = 0; j < res.width; j++) {
                if (res.boundaries[i * res.width + j] != 1) continue;

                if (j > 0 && j < res.width - 1) {
                    edgeSet.add(labelMatrix[i][j - 1] + "," + labelMatrix[i][j + 1]);
                }
                if (i > 0 && i < res.height - 1) {
                    edgeSet.add(labelMatrix[i - 1][j] + "," + labelMatrix[i + 1][j]);
                }
            }
        }

        double[][] x = new double[nodeFeatures.size()][7];
        double[][] pos = new double[nodeFeatures.size()][2];
        Map<Integer, Integer> labelToIndex = new HashMap<>();
        int index = 0;
        for (int label : nodeFeatures.keySet()) {
            labelToIndex.put(label, index);
            x[index] = nodeFeatures.get(label);
            pos[index] = Arrays.copyOf(nodeFeatures.get(label), 2);
            index++;
        }

        int[][] edgeIndex = new int[2][edgeSet.size()];
        int ei = 0;
        for (String edge : edgeSet) {
            String[] parts = edge.split(",");
            Integer u = labelToIndex.get(Integer.parseInt(parts[0]));
            Integer v = labelToIndex.get(Integer.parseInt(parts[1]));
	         // Skip if either label is missing (unregistered in node features)
            if (u == null || v == null)
                continue;
            edgeIndex[0][ei] = u;
            edgeIndex[1][ei] = v;
            ei++;
        }

        return new GraphResult(x, edgeIndex, pos, labelMatrix);
    }
}