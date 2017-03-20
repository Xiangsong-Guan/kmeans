package kmeans

import (
	"errors"
)

/*
This module provides common distance functions for measuring distance
between observations.
*/

// Cosine 余弦相似度
func Cosine(first, second Observation) (ret float64, err error) {
	topvalue := float64(0.0)

	for name, count := range first {
		_, ok := second[name]

		if ok {
			topvalue = topvalue + (count * second[name])
		}
	}

	mag := first.magnitude() * second.magnitude()

	if mag != 0.0 {
		return topvalue / mag, nil
	}
	return 0.0, errors.New("magnitude of vector is zero")
}
