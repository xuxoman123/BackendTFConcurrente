package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
)

type Labels struct {
	Nombre string `json:"nombre"`
	Cont   int    `json:"cont"`
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

//Punto es el punto
type Punto struct {
	X     float64 `json:"x"`     // x coordonate
	Y     float64 `json:"y"`     // y coordonate
	Label string  `json:"label"` // the classifier
}

type Block []Data

func (b Block) Len() int           { return len(b) }
func (b Block) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b Block) Less(i, j int) bool { return b[i].Distancia < b[j].Distancia }

//JSON_Input es el punto a recibir
type JSON_Input struct {
	X float64 `json:"x"` // x coordonate
	Y float64 `json:"y"` // y coordonate
	K byte    `json:"k"` // k value
}

type JSON_Output struct {
	Path  []Labels `json:"path"`
	Class string   `json:"class"`
}

var retorno JSON_Output

func LoadData() (data []Data, err error) {

	resp, err_http := http.Get("https://raw.githubusercontent.com/GoDie910/Concurrente_TA2_CSV/main/dataaa.csv")
	ValidError(err_http)
	defer resp.Body.Close()
	reader := csv.NewReader(resp.Body)
	reader.Comma = ','
	records, err := reader.ReadAll()

	filas := len(records)
	columnas := len(records[0])
	if columnas < 3 {
		return nil, fmt.Errorf("Cannot not load this data")
	}

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

func DEuclidiana(A Punto, X Punto) (distancia float64, err error) {
	distancia = math.Sqrt(math.Pow((X.X-A.X), 2) + math.Pow((X.Y-A.Y), 2))
	if distancia < 0 {
		return 0, fmt.Errorf("Invalid euclidian distance, the result is negative")
	}

	return distancia, nil
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

func Knn(data []Data, k byte, X *Punto) (err error) {
	n := len(data)

	for i := 0; i < n; i++ {
		if data[i].Distancia, err = DEuclidiana(data[i].Punto, *X); err != nil {
			return err
		}
	}

	var blk Block
	blk = data

	sort.Sort(blk)
	var save []Labels

	if int(k) > n {
		return nil
	}
	for i := byte(0); i < k; i++ {
		save = IncrementoLabels(data[i].Punto.Label, save)
	}

	fmt.Printf("[*] Using k as %d\n", k)
	fmt.Println()
	fmt.Printf("[*] %+v\n", save)
	fmt.Println()

	var deletable Labels
	deletable.Nombre = "deletable"
	deletable.Cont = 0
	save = append(save, deletable)
	copy(save[1:], save[0:(len(save)-1)])
	var K Labels
	K.Nombre = "k = " + strconv.Itoa(int(k))
	K.Cont = -1
	save[0] = K

	retorno.Path = save

	max := 0
	var maxLabel string
	m := len(save)
	for i := 0; i < m; i++ {
		if max < save[i].Cont {
			max = save[i].Cont
			maxLabel = save[i].Nombre
		}
	}

	X.Label = maxLabel
	retorno.Class = "K = " + strconv.Itoa(int(k)) + " | Class: " + maxLabel
	return nil
}

func worker(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Counter-Type", "application/json")

	data, err := LoadData()
	ValidError(err)

	var json_input JSON_Input
	_ = json.NewDecoder(r.Body).Decode(&json_input)
	var X Punto
	X.X = json_input.X
	X.Y = json_input.Y
	var k = json_input.K

	err = Knn(data, k, &X)

	ValidError(err)
	fmt.Printf("[*] Result for X is ")
	fmt.Println(X.Label)

	json.NewEncoder(w).Encode(retorno)
	var aux JSON_Output
	retorno = aux
}

func report(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Counter-Type", "application/json")

	json.NewEncoder(w).Encode("Worker 5 reportandose!")
}

func main() {
	//Init router
	r := mux.NewRouter()

	//Route Handlers / Endpoints
	r.HandleFunc("/api/knn/nodo_reportarse", report).Methods("GET")
	r.HandleFunc("/api/knn/nodo_llamar", worker).Methods("POST")

	//log.Fatal(http.ListenAndServe(":5005", r))
	log.Fatal(
		http.ListenAndServe(
			":5005",
			handlers.CORS(
				handlers.AllowedHeaders([]string{"X-Requested-With", "Content-Type", "Authorization"}),
				handlers.AllowedMethods([]string{"GET", "POST", "PUT", "HEAD", "OPTIONS"}),
				handlers.AllowedOrigins([]string{"*"}))(r)))
}
