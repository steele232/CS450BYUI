package main

import (
	"bytes"
	"io/ioutil"
	"log"

	"github.com/kniren/gota/dataframe"
	"github.com/kniren/gota/series"
)

func LoadIris() dataframe.DataFrame {

	filename := "./datasets/iris_binned.csv"

	csv, err := ioutil.ReadFile(filename) // returns []byte
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(csv), dataframe.HasHeader(false))

	return df
}

func LoadVoting() dataframe.DataFrame {

	filename := "datasets/house_voting_84_post.csv"

	csv, err := ioutil.ReadFile(filename) // returns []byte
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(csv), dataframe.HasHeader(false), dataframe.DefaultType(series.Bool))

	return df
}

func LoadAutoMpg() dataframe.DataFrame {

	filename := "datasets/auto_mpg_post.csv"

	csv, err := ioutil.ReadFile(filename) // returns []byte
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(csv), dataframe.HasHeader(false), dataframe.DetectTypes(true))

	return df
}
