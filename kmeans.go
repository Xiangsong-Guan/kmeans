package kmeans

import (
	"math"
	"math/rand"
)

// Observation Data Abstraction for an N-dimensional
// observation
type Observation map[string]float64

// ClusteredObservation Abstracts the Observation with a cluster number
// Update and computeation becomes more efficient
type ClusteredObservation struct {
	ClusterNumber int
	Observation
}

// DistanceFunction To compute the distanfe between observations
type DistanceFunction func(first, second Observation) (float64, error)

func (observation Observation) magnitude() float64 {
	total := 0.0

	for _, v := range observation {
		total = total + math.Pow(v, 2)
	}

	return math.Sqrt(total)
}

// Summation of two vectors
func (observation Observation) add(otherObservation Observation) {
	for ii, jj := range otherObservation {
		_, ok := observation[ii]
		if !ok {
			observation[ii] = jj
		}
		observation[ii] += jj
	}
}

// multiplication of a vector with a scalar
func (observation Observation) mul(scalar float64) {
	for ii := range observation {
		observation[ii] *= scalar
	}
}

// Near Find the closest observation and return the distance
// Index of observation, distance
func Near(p ClusteredObservation, mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance, _ := distanceFunction(p.Observation, mean[0])
	for i := 1; i < len(mean); i++ {
		squaredDistance, _ := distanceFunction(p.Observation, mean[i])
		if squaredDistance < minSquaredDistance {
			minSquaredDistance = squaredDistance
			indexOfCluster = i
		}
	}
	return indexOfCluster, math.Sqrt(minSquaredDistance)
}

// Instead of initializing randomly the seeds, make a sound decision of initializing
func seed(data []ClusteredObservation, k int, distanceFunction DistanceFunction) []Observation {
	s := make([]Observation, k)
	s[0] = data[rand.Intn(len(data))].Observation
	d2 := make([]float64, len(data))
	for ii := 1; ii < k; ii++ {
		var sum float64
		for jj, p := range data {
			_, dMin := Near(p, s[:ii], distanceFunction)
			d2[jj] = dMin * dMin
			sum += d2[jj]
		}
		target := rand.Float64() * sum
		jj := 0
		for sum = d2[0]; sum < target; sum += d2[jj] {
			jj++
		}
		s[ii] = data[jj].Observation
	}
	return s
}

// K-Means Algorithm
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) []ClusteredObservation {
	counter := 0
	for ii, jj := range data {
		closestCluster, _ := Near(jj, mean, distanceFunction)
		data[ii].ClusterNumber = closestCluster
	}
	mLen := make([]int, len(mean))
	for n := len(data[0].Observation); ; {
		for ii := range mean {
			mean[ii] = make(Observation, n)
			mLen[ii] = 0
		}
		for _, p := range data {
			mean[p.ClusterNumber].add(p.Observation)
			mLen[p.ClusterNumber]++
		}
		for ii := range mean {
			mean[ii].mul(1 / float64(mLen[ii]))
		}
		var changes int
		for ii, p := range data {
			if closestCluster, _ := Near(p, mean, distanceFunction); closestCluster != p.ClusterNumber {
				changes++
				data[ii].ClusterNumber = closestCluster
			}
		}
		counter++
		if changes == 0 || counter > threshold {
			return data
		}
	}
}

// Kmeans Algorithm with smart seeds as known as K-Means ++
func Kmeans(rawData []Observation, k int, distanceFunction DistanceFunction, threshold int) ([]int, []ClusteredObservation) {
	data := make([]ClusteredObservation, len(rawData))
	for ii, jj := range rawData {
		data[ii].Observation = jj
	}
	seeds := seed(data, k, distanceFunction)
	clusteredData := kmeans(data, seeds, distanceFunction, threshold)
	labels := make([]int, len(clusteredData))
	for ii, jj := range clusteredData {
		labels[ii] = jj.ClusterNumber
	}
	return labels, clusteredData
}
