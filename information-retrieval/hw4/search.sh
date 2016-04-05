#!/usr/bin/env bash

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

IFS=$"\n"
my_var=`cat /dev/stdin`

#echo $my_var

IFS=$" "

$SCRIPTPATH/search.py "$my_var"
