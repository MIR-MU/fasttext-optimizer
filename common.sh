#!/bin/bash

set -e
set -o pipefail

export LC_ALL=C

if (( $# < 1 ))
then
    printf 'Usage: %s LANGUAGE_ISO_CODE [LANGUAGE_ISO_CODE ...]\n' "$0"
    exit 1
fi

download_wikipedia_dump() {
    (
        set -e
        LANGUAGE="$1"
        echo Downloading "$LANGUAGE" wikipedia dump to data/wikimedia
        trap 'rm -rf data/wikimedia/*/' EXIT
        printf 'Downloading Wikipedia text to data/wikimedia/wiki.%s.txt\n' "$LANGUAGE"
        fastText/get-wikimedia.sh <<< "$LANGUAGE"$'\n'y
        trap '' EXIT
        cd data/wikimedia
        FILENAME="$(echo ./*/wiki."$LANGUAGE".txt)"
        mv "$FILENAME" .
        rm -rf "${FILENAME%/*}"
        cd -
        rm -rf data/wikimedia/*/
    )
}
