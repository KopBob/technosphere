#!/usr/bin/env bash

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

IFS=$"\n"
queries=`cat /dev/stdin`

IFS=$" "
$SCRIPTPATH/search.py "$queries"
