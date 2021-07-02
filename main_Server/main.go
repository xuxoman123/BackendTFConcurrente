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

type Worker_Request_Trained struct {
	X    float64 `json:"x"` // x coordonate
	Y    float64 `json:"y"` // y coordonate
	K    byte    `json:"k"`
	Data []Data  `json:"data"`
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

type Conteo struct {
	Clase        string
	repeticiones int
	K_mayor      int
}

type Data struct {
	Punto     Punto   `json:"punto"`     // point that has coordonates x, y and a label
	Distancia float64 `json:"distancia"` // the distance from X To point
	Anadido   string  `json:"tipo"`      //añadido
	Anadido2  string  `json:"fecha"`
}

func (d Data) String() string {
	return fmt.Sprintf(
		"X = %f Y = %f, Distance = %f, Label = %s, Tipo = %s, Fecha = %s\n",
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

	fmt.Println("\n[*] Loading records of Github")
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
		data[i-1].Anadido = records[i][3]
		data[i-1].Anadido2 = records[i][4]

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

func train_AI(X *Punto) {
	fmt.Println("\n==================================")
	fmt.Println("==================================")
	fmt.Println("TRAINING AI")

	var conteos []Conteo

	for i := range retorno.Classes {
		var conteo Conteo

		splits := strings.Split(retorno.Classes[i], " | Class: ")
		conteo.Clase = splits[1]
		splits = strings.Split(splits[0], " = ")
		K_mayor, err := strconv.Atoi(splits[1])
		ValidError(err)
		conteo.K_mayor = K_mayor
		conteo.repeticiones = 1

		exist := false

		for j := range conteos {
			if conteos[j].Clase == conteo.Clase {
				conteos[j].repeticiones = conteos[j].repeticiones + 1
				if conteos[j].K_mayor < conteo.K_mayor {
					conteos[j].K_mayor = conteo.K_mayor
				}
				exist = true
				break
			}
		}
		if exist == false {
			conteos = append(conteos, conteo)
		}
	}

	fmt.Println("\nConteos: ", conteos)

	var clase_final Conteo
	for i := range conteos {
		if i == 0 {
			clase_final.Clase = conteos[0].Clase
			clase_final.repeticiones = conteos[0].repeticiones
			clase_final.K_mayor = conteos[0].K_mayor
		} else if clase_final.repeticiones > conteos[i].repeticiones {
			continue
		} else if clase_final.repeticiones < conteos[i].repeticiones {
			clase_final.Clase = conteos[i].Clase
			clase_final.repeticiones = conteos[i].repeticiones
			clase_final.K_mayor = conteos[i].K_mayor
		} else if clase_final.K_mayor < conteos[i].K_mayor {
			clase_final.Clase = conteos[i].Clase
			clase_final.repeticiones = conteos[i].repeticiones
			clase_final.K_mayor = conteos[i].K_mayor
		}
	}

	fmt.Println("\nClase Final: ", clase_final)

	data_trained_save(X, clase_final.Clase)
}

func data_trained_save(X *Punto, clase string) {
	file, err := os.OpenFile("data_trained.csv", os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0600)
	ValidError(err)

	defer file.Close()

	//writer := csv.NewWriter(file)

	var data = "\n" + strconv.FormatFloat(X.X, 'f', 5, 64) + "," + strconv.FormatFloat(X.Y, 'f', 5, 64) + "," + clase + ",1" + ",1"

	//err = writer.WriteAll(data)
	_, err = file.WriteString(data)
	ValidError(err)
}

func data_trained_read() (data []Data, err error) {
	file, err := os.Open("data_trained.csv")
	ValidError(err)

	reader := csv.NewReader(file)
	reader.Comma = ','
	records, err := reader.ReadAll()
	ValidError(err)

	fmt.Println("\n[*] Loading Trained data")
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
		data[i-1].Anadido = records[i][3]
		data[i-1].Anadido2 = records[i][4]

	}
	return data, nil
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

	fmt.Println("\nWorker: ", id_worker, " | K: ", k, "PATH | ", body.Path, "CLASS | ", body.Class)

	defer wg.Done()
}

func call_worker_trained(data *[]Data, i int, k byte, X *Punto, wg *sync.WaitGroup) {

	id_worker := (i % 5) + 1
	var url_call string

	switch id_worker {
	case 1:
		url_call = "http://localhost:5001/api/knn/nodo_llamar_entrenado"
	case 2:
		url_call = "http://localhost:5002/api/knn/nodo_llamar_entrenado"
	case 3:
		url_call = "http://localhost:5003/api/knn/nodo_llamar_entrenado"
	case 4:
		url_call = "http://localhost:5004/api/knn/nodo_llamar_entrenado"
	case 5:
		url_call = "http://localhost:5005/api/knn/nodo_llamar_entrenado"
	}

	req := Worker_Request_Trained{X.X, X.Y, k, *data}
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

	fmt.Println("\nWorker: ", id_worker, " | K: ", k, "PATH | ", body.Path, "CLASS | ", body.Class)

	defer wg.Done()
}

func API_KNN(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Counter-Type", "application/json")

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
			fmt.Println()
			fmt.Println("[*] Loading Euclidian table")
			fmt.Println()
			fmt.Println(data)
			retorno.Data = data
		}
		go call_worker(i, k[i], &X, &wg)
	}

	wg.Wait()

	train_AI(&X)

	json.NewEncoder(w).Encode(retorno)
	var aux JSON_Output
	retorno = aux
}

func API_KNN_entrenado(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Counter-Type", "application/json")

	data, err := data_trained_read()
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
			fmt.Println()
			fmt.Println("[*] Loading Euclidian table")
			fmt.Println()
			fmt.Println(data)
			retorno.Data = data
		}
		go call_worker_trained(&data, i, k[i], &X, &wg)
	}

	wg.Wait()

	json.NewEncoder(w).Encode(retorno)
	var aux JSON_Output
	retorno = aux
}

func main() {
	//Init router
	r := mux.NewRouter()

	//Route Handlers / Endpoints
	r.HandleFunc("/api/knn", API_KNN).Methods("POST")
	r.HandleFunc("/api/knn_entrenado", API_KNN_entrenado).Methods("POST")

	//log.Fatal(http.ListenAndServe(":5000", r))
	log.Fatal(
		http.ListenAndServe(
			":5000",
			handlers.CORS(
				handlers.AllowedHeaders([]string{"X-Requested-With", "Content-Type", "Authorization"}),
				handlers.AllowedMethods([]string{"GET", "POST", "PUT", "HEAD", "OPTIONS"}),
				handlers.AllowedOrigins([]string{"*"}))(r)))
}
