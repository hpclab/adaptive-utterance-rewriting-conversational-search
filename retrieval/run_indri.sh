#!/bin/sh

input = $1
output = $2

echo $input

./IndriRunQuery par_query_lm_NOSW_PRF20.indri $input 2> err > $output

echo "done"