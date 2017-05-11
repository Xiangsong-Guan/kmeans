package kmeans

import "math"

/*
This module provides common distance functions for measuring distance
between observations.
*/

// EuclideanDistance 欧氏距离
func EuclideanDistance(firstVector, secondVector Observation) float64 {
	distance := 0.
	for ii := range firstVector {
		distance += (firstVector[ii] - secondVector[ii]) * (firstVector[ii] - secondVector[ii])
	}
	return math.Sqrt(distance)
}

// Cosine 余弦相似度
func Cosine(first, second Observation) float64 {
	topvalue := 0.0
	for i, v := range first {
		topvalue = topvalue + (v * second[i])
	}
	return topvalue / (first.Magnitude() * second.Magnitude())
}
