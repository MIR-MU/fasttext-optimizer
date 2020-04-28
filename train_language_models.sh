#!/bin/bash

source ./common.sh

FASTTEXT_MODEL_DIRECTORY="$PWD"/data/wikimedia
LANGUAGE_MODEL_DIRECTORY="$PWD"/data/wmt13
SCRIPT_DIRECTORY=languageModels
LANGUAGE_MODEL_PARAMETERS=(--cuda --tied --emsize 100 --nhid 100 --nlayers 1 --dropout 0 --epochs 1)

cd "$SCRIPT_DIRECTORY"

if [[ ! -e "$LANGUAGE_MODEL_DIRECTORY" ]]
then
    (
        cd -
        download_wmt13
    )
fi

for LANGUAGE
do

    for NGRAM_SIZES in default suggested
    do

        echo Training "$LANGUAGE" language model with "$NGRAM_SIZES" n-gram sizes

        BATCH_SIZE=10
        while (( BATCH_SIZE > 0 ))
        do

            FASTTEXT_MODEL_BASENAME="$FASTTEXT_MODEL_DIRECTORY"/wiki."$LANGUAGE"."$NGRAM_SIZES"
            FASTTEXT_MODEL_FILENAME="$FASTTEXT_MODEL_BASENAME".bin
            LANGUAGE_MODEL_BASENAME="$LANGUAGE_MODEL_DIRECTORY"/wiki."$LANGUAGE".lm."$NGRAM_SIZES"
            LANGUAGE_MODEL_LOG_FILENAME="$LANGUAGE_MODEL_BASENAME".log.gz
            LANGUAGE_MODEL_FILENAME="$LANGUAGE_MODEL_BASENAME".pt
            CURRENT_LANGUAGE_MODEL_PARAMETERS=("${LANGUAGE_MODEL_PARAMETERS[@]}" --batch_size "$BATCH_SIZE" --language "$LANGUAGE" --save "${LANGUAGE_MODEL_FILENAME}" --fasttext_model "$FASTTEXT_MODEL_FILENAME")
            if [[ ! -e "$FASTTEXT_MODEL_FILENAME" ]]
            then
                (
                    cd -
                    ./train_fasttext_models.sh "$LANGUAGE"
                )
            fi

            if [[ ! -e "$LANGUAGE_MODEL_LOG_FILENAME" || ! -e "$LANGUAGE_MODEL_FILENAME" ]]
            then
                if (
                    echo Using batch size $BATCH_SIZE
                    time python3 -u -m main "${CURRENT_LANGUAGE_MODEL_PARAMETERS[@]}"
                ) |& tee >(gzip > "$LANGUAGE_MODEL_LOG_FILENAME")
                then
                    break
                fi
            else
                break
            fi

            (( BATCH_SIZE -= 1 ))

        done

        if (( BATCH_SIZE = 0 ))
        then
            echo Failed to train "$LANGUAGE_MODEL_FILENAME", consult "$LANGUAGE_MODEL_LOG_FILENAME" 2>&1
            exit 1
        fi

    done

done
