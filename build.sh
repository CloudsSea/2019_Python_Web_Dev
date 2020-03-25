#!/usr/bin/env bash

cd www
tar_file='dist-awesome.tar.gz'
includes="static templates /opt/install/docker/dataanalysis/project/intelligent"
excludes=("test" ".*" "*.pyc" "*.pyo")


rm -f dist/$tar_file
#tar --dereference -czvf ../dist/$tar_file  --exclude=test --exclude=.* --exclude=*.pyc --exclude=*.pyo $includes *.py *.txt
tar -czvf ../dist/$tar_file $includes * 
