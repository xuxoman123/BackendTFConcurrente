package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strings"
	"sync"

	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
)

//JSON_Input es el punto a recibir
type JSON_Input struct {
	X float64 `json:"x"` // x coordonate
	Y float64 `json:"y"` // y coordonate
	K []byte  `json:"k"`
}

type JSON_Output struct {
	Data    []Data     `json:"data"`
	Paths   [][]Labels `json:"paths"`
	Classes []string   `json:"classes"`
}

type Worker_Request struct {
	X float64 `json:"x"` // x coordonate
	Y float64 `json:"y"` // y coordonate
	K byte    `json:"k"`
}

type Worker_Response struct {
	Path  []Labels `json:"path"`
	Class string   `json:"class"`
}

var retorno JSON_Output

//Punto es el punto
type Punto struct {
	X     float64 `json:"x"`     // x coordonate
	Y     float64 `json:"y"`     // y coordonate
	Label string  `json:"label"` // the classifier
}

type Labels struct {
	Nombre string `json:"nombre"`
	Cont   int    `json:"cont"`
}

func (p Punto) String() string {
	return fmt.Sprintf("[*] X = %f, Y = %f Label = %s\n", p.X, p.Y, p.Label)
}

type Data struct {
	Punto     Punto   `json:"punto"`     // point that has coordonates x, y and a label
	Distancia float64 `json:"distancia"` // the distance from X To point
	Anadido   float64 `json:"anadido"`   //añadido
	Anadido2  float64 `json:"anadido2"`
}

func (d Data) String() string {
	return fmt.Sprintf(
		"X = %f Y = %f, Distance = %f Label = %s, Extra1 = %f, Extra2 = %f\n",
		d.Punto.X, d.Punto.Y, d.Distancia, d.Punto.Label, d.Anadido, d.Anadido2,
	)
}

type Block []Data

func (b Block) Len() int           { return len(b) }
func (b Block) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b Block) Less(i, j int) bool { return b[i].Distancia < b[j].Distancia }

func DEuclidiana(A Punto, X Punto) (distancia float64, err error) {
	distancia = math.Sqrt(math.Pow((X.X-A.X), 2) + math.Pow((X.Y-A.Y), 2))
	if distancia < 0 {
		return 0, fmt.Errorf("Invalid euclidian distance, the result is negative")
	}

	return distancia, nil
}

func LoadData() (data []Data, err error) {

	resp, err_http := http.Get("https://raw.githubusercontent.com/GoDie910/Concurrente_TA2_CSV/main/dataaa.csv")
	ValidError(err_http)
	defer resp.Body.Close()
	reader := csv.NewReader(resp.Body)
	reader.Comma = ','
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
		data[i-1].Punto.X = value
		value, err = strconv.ParseFloat(records[i][1], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse Y value: %v", err)
		}
		data[i-1].Punto.Y = value
		data[i-1].Punto.Label = records[i][2]

		value, err = strconv.ParseFloat(records[i][3], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse E1 value: %v", err)
		}
		data[i-1].Anadido = value

		value, err = strconv.ParseFloat(records[i][4], 64)
		if err != nil {
			return nil, fmt.Errorf("cannot parse E2 value: %v", err)
		}
		data[i-1].Anadido2 = value

	}
	return data, nil
}

func ValidError(err error) {
	if err != nil {
		fmt.Printf("[!] %s\n", err.Error())
		os.Exit(1)
	}
}

func Knn(data []Data, X *Punto) (err error) {
	n := len(data)
	// compute every point distance with X
	for i := 0; i < n; i++ {
		if data[i].Distancia, err = DEuclidiana(data[i].Punto, *X); err != nil {
			return err
		}
	}

	var blk Block
	blk = data
	// sort the data in ascending order order
	sort.Sort(blk)

	return nil
}

func IncrementoLabels(label string, labels []Labels) []Labels {
	if labels == nil {
		labels = append(labels, Labels{
			Nombre: label,
			Cont:   1,
		})
		return labels
	}

	cont := len(labels)
	for i := 0; i < cont; i++ {
		if strings.Compare(labels[i].Nombre, label) == 0 {
			labels[i].Cont++
			return labels
		}
	}

	return append(labels, Labels{
		Nombre: label,
		Cont:   1,
	})
}

func API_KNN(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Counter-Type", "application/json")

	//data, err := LoadData(resp.Body)
	data, err := LoadData()
	ValidError(err)

	// read from JSON
	var json_input JSON_Input
	_ = json.NewDecoder(r.Body).Decode(&json_input)
	var X Punto
	X.X = json_input.X
	X.Y = json_input.Y
	var k = json_input.K

	n := len(k)

	err = Knn(data, &X)
	ValidError(err)

	var wg sync.WaitGroup
	wg.Add(len(k))

	for i := 0; i < n; i++ {
		if i == 0 {
			fmt.Println(data)
			retorno.Data = data
		}
		go call_worker(i, k[i], &X, &wg)
	}

	wg.Wait()

	json.NewEncoder(w).Encode(retorno)
	var aux JSON_Output
	retorno = aux
}

func call_worker(i int, k byte, X *Punto, wg *sync.WaitGroup) {

	id_worker := (i % 5) + 1
	var url_call string

	switch id_worker {
	case 1:
		url_call = "http://localhost:5001/api/knn/nodo_llamar"
	case 2:
		url_call = "http://localhost:5002/api/knn/nodo_llamar"
	case 3:
		url_call = "http://localhost:5003/api/knn/nodo_llamar"
	case 4:
		url_call = "http://localhost:5004/api/knn/nodo_llamar"
	case 5:
		url_call = "http://localhost:5005/api/knn/nodo_llamar"
	}

	req := Worker_Request{X.X, X.Y, k}
	jsonReq, err := json.Marshal(req)
	ValidError(err)

	resp, err := http.Post(url_call, "application/json; charset=utf-8", bytes.NewBuffer(jsonReq))
	ValidError(err)

	defer resp.Body.Close()
	bodyBytes, _ := ioutil.ReadAll(resp.Body)
	var body Worker_Response
	json.Unmarshal(bodyBytes, &body)

	retorno.Paths = append(retorno.Paths, body.Path)
	retorno.Classes = append(retorno.Classes, body.Class)

	fmt.Println()
	fmt.Println("Worker: ", id_worker, " | K: ", k, "PATH | ", body.Path, "CLASS | ", body.Class)

	defer wg.Done()

}

func main() {
	//Init router
	r := mux.NewRouter()

	//Route Handlers / Endpoints
	r.HandleFunc("/api/knn", API_KNN).Methods("POST")

	//log.Fatal(http.ListenAndServe(":5000", r))
	log.Fatal(
		http.ListenAndServe(
			":5000",
			handlers.CORS(
				handlers.AllowedHeaders([]string{"X-Requested-With", "Content-Type", "Authorization"}),
				handlers.AllowedMethods([]string{"GET", "POST", "PUT", "HEAD", "OPTIONS"}),
				handlers.AllowedOrigins([]string{"*"}))(r)))
}
