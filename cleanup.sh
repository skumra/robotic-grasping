#!/usr/bin/env bash

find logs/ -maxdepth 1 -type d | grep -v ^\\.$ | xargs -n 1 du -s | while read size name ; do if [ $size -le 10485 ] ; then echo rm -rf $name ; fi done
