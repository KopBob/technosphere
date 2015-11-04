#! /bin/bash




INPUT=(
    "10 10 10"
    "9 10 10"
    "12 9 AB"
    "21 9 ab"
    "21 9 AZB"
    "21 9 312321312312"
    "10 2 A0B0"
)
OUTPUT=(
    "error"
    "error"
    "155"
    "265"
    "error"
    "5165706746868025"
    "error"
)

INPUT_LEN=${#INPUT[@]}
OUTPUT_LEN=${#OUTPUT[@]}

if ! [ "$INPUT_LEN" -eq "$OUTPUT_LEN" ]; then
    echo "Len of INPUT and OUTPUT not the same."
    exit 1
fi

for i in $(seq $INPUT_LEN $END); do
    echo -ne "${INPUT[i-1]} | ${OUTPUT[i-1]} --> "
    echo "${INPUT[i-1]}" | ./a.out | grep -Fq "${OUTPUT[i-1]}" && echo "SUCCESS" || echo "FAIL"
done

# echo "$in" | ./a.out | grep -Fq "$res" && echo "SUCCESS" || echo "FAIL"


