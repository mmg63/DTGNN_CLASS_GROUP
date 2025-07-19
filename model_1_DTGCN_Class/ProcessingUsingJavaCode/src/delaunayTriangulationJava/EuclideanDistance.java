package delaunayTriangulationJava;

public class EuclideanDistance{
	public static double calculate(double[] coord1, double[] coord2) {
		return Math.sqrt(Math.pow(coord1[0] - coord2[0], 2) + Math.pow(coord1[0] - coord2[1], 2));
	}
}