#! /bin/bash




INPUT=(
    "()"
    "(0)"
    "(1+2)/3"
    "1/0"
    "1/1"
    "1/1*10-1"
    "-(-1)*(-1)"
)
OUTPUT=(
    "error"
    "0.00"
    "1.00"
    "error"
    "1.00"
    "9.00"
    "-1.00"
)

INPUT_LEN=${#INPUT[@]}
OUTPUT_LEN=${#OUTPUT[@]}

if ! [ "$INPUT_LEN" -eq "$OUTPUT_LEN" ]; then
    echo "Len of INPUT and OUTPUT not the same."
    exit 1
fi

for i in $(seq $INPUT_LEN $END); do
    echo -ne "${INPUT[i-1]} | ${OUTPUT[i-1]} --> "
    echo -e "${INPUT[i-1]}" | ./a.out | grep -Fq -- "${OUTPUT[i-1]}" && echo "SUCCESS" || echo "FAIL"
done

# echo "$in" | ./a.out | grep -Fq "$res" && echo "SUCCESS" || echo "FAIL"


