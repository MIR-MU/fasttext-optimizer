#!/bin/bash

source ./common.sh

FASTTEXT_DIRECTORY=fastText
FASTTEXT="$FASTTEXT_DIRECTORY"/fasttext
FASTTEXT_PARAMETERS=(skipgram -bucket 2000000 -dim 300 -epoch 5 -loss ns -lr 0.05 -minCount 5 -neg 5 -t 0.001 -thread 80 -ws 5)
DEFAULT_NGRAM_SIZES=(-minn 3 -maxn 6)
OUTPUT_DIRECTORY=data/wikimedia

if [[ ! -e "$FASTTEXT" ]]
then
    echo Compiling fastText
    make -C "$FASTTEXT_DIRECTORY"
fi

for LANGUAGE
do

    if [[ ! -e "$OUTPUT_DIRECTORY"/wiki."$LANGUAGE".txt ]]
    then
        download_wikipedia_dump "$LANGUAGE"
    fi

    read -r -a SUGGESTED_NGRAM_SIZES <<< "$(./suggest_subword_sizes.sh "$LANGUAGE" | grep '^Suggested subword sizes' | sed -r 's/.*(-minn [0-9]* -maxn [0-9]*).*/\1/')"
    INPUT=(-input "$OUTPUT_DIRECTORY"/wiki."$LANGUAGE".txt)

    OUTPUT_BASENAME="$OUTPUT_DIRECTORY"/wiki."$LANGUAGE".default
    LOG_FILENAME="$OUTPUT_BASENAME".log.gz
    INPUT_OUTPUT=("${INPUT[@]}" -output "$OUTPUT_BASENAME")
    if [[ ! -e "$OUTPUT_BASENAME".vec ]]
    then
        echo Training "$LANGUAGE" fastText model with default n-gram sizes "(${DEFAULT_NGRAM_SIZES[*]})"
        trap 'rm "$LOG_FILENAME"' EXIT
        "$FASTTEXT" "${FASTTEXT_PARAMETERS[@]}" "${INPUT_OUTPUT[@]}" "${DEFAULT_NGRAM_SIZES[@]}"   |& tee >(gzip > "$LOG_FILENAME")
        trap '' EXIT
    fi

    OUTPUT_BASENAME="$OUTPUT_DIRECTORY"/wiki."$LANGUAGE".suggested
    LOG_FILENAME="$OUTPUT_BASENAME".log.gz
    INPUT_OUTPUT=("${INPUT[@]}" -output "$OUTPUT_BASENAME")
    if [[ ! -e "$OUTPUT_BASENAME".vec ]]
    then
        trap 'rm "$LOG_FILENAME"' EXIT
        echo Training "$LANGUAGE" fastText model with suggested n-gram sizes "(${SUGGESTED_NGRAM_SIZES[*]})"
        "$FASTTEXT" "${FASTTEXT_PARAMETERS[@]}" "${INPUT_OUTPUT[@]}" "${SUGGESTED_NGRAM_SIZES[@]}" |& tee >(gzip > "$LOG_FILENAME")
        trap '' EXIT
    fi

done
