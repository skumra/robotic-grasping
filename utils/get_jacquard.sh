#!/usr/bin/env bash

END=11
for ((i=0;i<=END;i++)); do
  wget --retry 5 https://jacquard.liris.cnrs.fr/data/Download/Jacquard_Dataset_$i.zip
  unzip Jacquard_Dataset_$i.zip
  rm Jacquard_Dataset_$i.zip
done