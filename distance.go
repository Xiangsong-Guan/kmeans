package kmeans

/*
This module provides common distance functions for measuring distance
between observations.
*/

// Cosine 余弦相似度
func Cosine(first, second Observation) float64 {
	topvalue := 0.0
	for i, v := range first {
		topvalue = topvalue + (v * second[i])
	}
	return topvalue / (first.Magnitude() * second.Magnitude())
}
