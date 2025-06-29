
#! /bin/bash
root_path=$(pwd)
build_path=$root_path/mybuild

if [ -e mybuild ]
then
    cd mybuild
    rm -r ./*
else
    mkdir mybuild
    cd mybuild
fi

cd $build_path
cmake -Dplatform=x86 ..
make -j3

