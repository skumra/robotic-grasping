#!/usr/bin/env bash

END=11
for ((i=0;i<=END;i++)); do
  curl -O --retry 5 --max-time 15 https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_$i.zip
  unzip Jacquard_Dataset_$i.zip
  rm Jacquard_Dataset_$i.zip
done