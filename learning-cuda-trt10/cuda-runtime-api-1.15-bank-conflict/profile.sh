#!/bin/bash
nsys profile --stats=true --export=none build/cudaTest
rm *.qdrep *.sqlite