#!/bin/bash

inputpoints=$1

label="set xlabel 'k2V';"
label+="set ylabel 'kl';"
label+="set zlabel 'kV';"
gpcommand="$label;splot '$inputpoints' with points pointtype 7 pointsize 0.3 notitle;"
gpcommand+="pause -1;"

gnuplot -e "$gpcommand"
