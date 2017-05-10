package kmeans

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"gopkg.in/cheggaaa/pb.v1"
)

// Observation Data Abstraction for an N-dimensional
// observation
type Observation map[int]float64

// ClusteredObservation Abstracts the Observation with a cluster number
// Update and computeation becomes more efficient
type ClusteredObservation struct {
	ClusterNumber int
	Observation
}

// DistanceFunction To compute the distanfe between observations
type DistanceFunction func(first, second Observation) float64

//Magnitude 模
func (observation Observation) Magnitude() float64 {
	total := 0.0
	for _, v := range observation {
		total = total + math.Pow(v, 2)
	}
	return math.Sqrt(total)
}

//Add Summation of two vectors
func (observation Observation) Add(otherObservation Observation) {
	for ii, jj := range otherObservation {
		observation[ii] += jj
	}
}

//Mul Multiplication of a vector with a scalar
func (observation Observation) Mul(scalar float64) {
	for ii := range observation {
		observation[ii] *= scalar
	}
}

// Near Find the closest observation and return the distance
// Index of observation, distance
func Near(p ClusteredObservation, mean []Observation, distanceFunction DistanceFunction) (int, float64) {
	indexOfCluster := 0
	minSquaredDistance := distanceFunction(p.Observation, mean[0])
	for i := 1; i < len(mean); i++ {
		squaredDistance := distanceFunction(p.Observation, mean[i])
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
func kmeans(data []ClusteredObservation, mean []Observation, distanceFunction DistanceFunction, threshold int) ([]ClusteredObservation, []Observation) {
	counter := 0
	for ii, jj := range data {
		closestCluster, _ := Near(jj, mean, distanceFunction)
		data[ii].ClusterNumber = closestCluster
	}
	mLen := make([]int, len(mean))

	//muti-thread init
	workers := runtime.NumCPU()
	tasksNum := len(data) / workers
	last := len(data) % workers
	meanLockers := make([]sync.Mutex, len(mean))
	done := make(chan bool, workers+1)
	runtime.GOMAXPROCS(runtime.NumCPU())

	//new bar
	bar := pb.StartNew(threshold)

	for {
		for ii := range mean {
			mean[ii] = make(Observation)
			mLen[ii] = 0
		}

		//muti-thread 分派任务1
		if last != 0 {
			go kmeansWorker1(data[0:last], mean, mLen, meanLockers, done)
		} else {
			done <- true
		}
		for i := 0; i < workers; i++ {
			go kmeansWorker1(data[last+i*tasksNum:last+(i+1)*tasksNum], mean, mLen, meanLockers, done)
		}
		for i := workers + 1; i > 0; {
			<-done
			i--
		}
		//for _, p := range data {
		//  mean[p.ClusterNumber].Add(p.Observation)
		//	mLen[p.ClusterNumber]++
		//}

		for ii := range mean {
			mean[ii].Mul(1 / float64(mLen[ii]))
		}

		//muti-thread 分派任务2
		changes := 0
		if last != 0 {
			go kmeansWorker2(data[0:last], mean, done)
		} else {
			done <- false
		}
		for i := 0; i < workers; i++ {
			go kmeansWorker2(data[last+i*tasksNum:last+(i+1)*tasksNum], mean, done)
		}
		for i := workers + 1; i > 0; {
			if res := <-done; !res {
				changes++
			}
			i--
		}
		//for ii, p := range data {
		//	if closestCluster, _ := Near(p, mean, distanceFunction); closestCluster != p.ClusterNumber {
		//		changes++
		//		data[ii].ClusterNumber = closestCluster
		//	}
		//}

		counter++
		if (changes == (workers + 1)) || (counter > threshold) {
			bar.Finish()
			return data, mean
		}
		bar.Increment()
	}
}

// Kmeans Algorithm with smart seeds as known as K-Means ++
func Kmeans(rawData []Observation, k int, distanceFunction DistanceFunction, threshold int) ([][]int, []Observation) {
	data := make([]ClusteredObservation, len(rawData))
	for ii, jj := range rawData {
		data[ii].Observation = jj
	}
	seeds := seed(data, k, distanceFunction)
	clusteredData, means := kmeans(data, seeds, distanceFunction, threshold)
	label := make([][]int, k)
	for ii, jj := range clusteredData {
		label[jj.ClusterNumber] = append(label[jj.ClusterNumber], ii)
	}
	return label, means
}
