package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

type Punto struct {
	x     float64 // x coordonate
	y     float64 // y coordonate
	label string  
}

type Labels struct {
	nombre string
	cont   int
}

func (p Punto) String() string {
	return fmt.Sprintf("[*] X = %f, Y = %f Label = %s, \n", p.x, p.y, p.label)
}

type Extras struct{
	extra1 float64//extra1
	extra2 string//extra2
}

type Data struct {
	punto     Punto   
	distancia float64
	anadido float64 //añadido
	anadido2 float64
}


func (d Data) String() string {
	return fmt.Sprintf(
		"X = %f Y = %f, Distance = %f Label = %s, Extra1 = %f, Extra2 = %f\n",
		d.punto.x, d.punto.y, d.distancia, d.punto.label, d.anadido, d.anadido2,
	)
}


type Block []Data

func (b Block) Len() int           { return len(b) }
func (b Block) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b Block) Less(i, j int) bool { return b[i].distancia < b[j].distancia }

func DEuclidiana(A Punto, X Punto) (distancia float64, err error) {
	distancia = math.Sqrt(math.Pow((X.x-A.x), 2) + math.Pow((X.y-A.y), 2))
	if distancia < 0 {
		return 0, fmt.Errorf("Invalid euclidian distance, the result is negative")
	}

	return distancia, nil
}

func LoadData(csvPath string) (data []Data, err error) {

	fd, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}

	reader := csv.NewReader(fd)

	records, err := reader.ReadAll()
	fmt.Println("[*] Loading records")
	fmt.Println()
	
	filas := len(records)
	columnas := len(records[0])
	if columnas < 3 {
		return nil, fmt.Errorf("Cannot not load this data")
	}
	for i := 0; i < filas; i++ {
		for j := 0; j < columnas; j++ {
			fmt.Printf("%s\t  ", records[i][j])
		}
		if i == 0 {
			fmt.Println()
		}
		fmt.Println()
	}
	fmt.Println()

	var value float64
	data = make([]Data, filas-1, filas-1)

	for i := 1; i < filas; i++ {
		value, err = strconv.ParseFloat(records[i][0], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse X value: %v", err)
		}
		data[i-1].punto.x = value
		value, err = strconv.ParseFloat(records[i][1], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse Y value: %v", err)
		}
		data[i-1].punto.y = value
		data[i-1].punto.label = records[i][2]


	
		value, err = strconv.ParseFloat(records[i][3], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse Y value: %v", err)
		}
		data[i-1].anadido= value

		value, err = strconv.ParseFloat(records[i][4], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse Y value: %v", err)
		}
		data[i-1].anadido2= value


	}
	return data, nil
}


func ValidError(err error) {
	if err != nil {
		fmt.Printf("[!] %s\n", err.Error())
		os.Exit(1)
	}
}


func Knn(data []Data, k byte, X *Punto) (err error) {
	n := len(data)

	for i := 0; i < n; i++ {
		if data[i].distancia, err = DEuclidiana(data[i].punto, *X); err != nil {
			return err
		}
	}

	var blk Block
	blk = data

	sort.Sort(blk)
	var save []Labels
	// pass
	if int(k) > n {
		return nil
	}
	for i := byte(0); i < k; i++ {
		save = IncrementoLabels(data[i].punto.label, save)
	}

	fmt.Printf("[*] Using k as %d\n", k)
	fmt.Println()
	fmt.Printf("[*] %+v\n", save)
	fmt.Println()

	max := 0
	var maxLabel string
	m := len(save)
	for i := 0; i < m; i++ {
		if max < save[i].cont {
			max = save[i].cont
			maxLabel = save[i].nombre
		}
	}

	X.label = maxLabel
	return nil
}


func IncrementoLabels(label string, labels []Labels) []Labels {
	if labels == nil {
		labels = append(labels, Labels{
			nombre: label,
			cont:   1,
		})
		return labels
	}

	cont := len(labels)
	for i := 0; i < cont; i++ {
		if strings.Compare(labels[i].nombre, label) == 0 {
			labels[i].cont++
			return labels
		}
	}

	return append(labels, Labels{
		nombre: label,
		cont:   1,
	})
}

func main() {
	fmt.Println()
	fmt.Println("[*] KNN algorithm")

	k := []byte{1, 3, 5, 7, 9, 11}

	data, err := LoadData("data.csv")
	ValidError(err)


	var X Punto
	fmt.Println("[i] Give the X point to look for")
	fmt.Printf("x = ")
	fmt.Scanf("%f", &X.x)
	fmt.Printf("\ny = ")
	fmt.Scanf("%f", &X.y)
	fmt.Println()

	n := len(k)
	for i := 0; i < n; i++ {
		err = Knn(data, k[i], &X)
		if i == 0 {
			fmt.Println(data)
		}
		ValidError(err)
		fmt.Printf("[*] Result for X is ")
		fmt.Println(X)
	}
}
