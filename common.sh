set -e
set -o pipefail

export LC_ALL=C

if [[ $# < 1 ]]
then
    printf 'Usage: %s LANGUAGE_ISO_CODE [LANGUAGE_ISO_CODE ...]\n' "$0"
    exit 1
fi

download_wmt13() {
    (
        set -e
        trap 'rm -rf ../data/wmt13/' EXIT
        echo Downloading wmt13 dataset to data/wmt13
        mkdir -p ../data/wmt13
        wget -q --show-progress 'http://www.statmt.org/wmt13/training-monolingual-news-2011.tgz' -O - | tar xzC ../data/wmt13/
        wget -q --show-progress 'http://www.statmt.org/wmt13/training-monolingual-nc-v8.tgz'     -O - | tar xzC ../data/wmt13/
        wget -q --show-progress 'http://www.statmt.org/wmt13/training-monolingual-news-2012.tgz' -O - | tar xzC ../data/wmt13/
        trap '' EXIT
    )
}
