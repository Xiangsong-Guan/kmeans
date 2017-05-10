package kmeans

import (
	"sync"
)

func kmeansWorker1(data []ClusteredObservation, mean []Observation, mLen []int, meanLockers []sync.Mutex, done chan<- bool) {
	for _, v := range data {
		num := v.ClusterNumber
		meanLockers[num].Lock()
		mean[num].Add(v.Observation)
		mLen[num]++
		meanLockers[num].Unlock()
	}
	done <- true
}

func kmeansWorker2(data []ClusteredObservation, mean []Observation, done chan<- bool) {
	changes := 0
	for i, v := range data {
		if closestCluster, _ := Near(v, mean, Cosine); closestCluster != v.ClusterNumber {
			data[i].ClusterNumber = closestCluster
			changes++
		}
	}
	if changes > 0 {
		done <- true
	} else {
		done <- false
	}
}
