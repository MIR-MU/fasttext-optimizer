#!/bin/bash

source ./common.sh

for LANGUAGE
do
    if [[ ! -e data/wikimedia/wiki.$LANGUAGE.txt && ! -e data/wikimedia/wiki.$LANGUAGE.json ]]
    then
        download_wikipedia_dump "$LANGUAGE"
    fi

    if [[ ! -e data/wikimedia/wiki.$LANGUAGE.json ]]
    then
        trap "rm data/wikimedia/wiki.$LANGUAGE.json" EXIT
        printf 'Saving subterm frequency statistics to data/wikimedia/wiki.%s.json\n' $LANGUAGE
        python3 -c '
from itertools import combinations
from sys import argv, stdout

from gensim.utils import simple_preprocess
from tqdm import tqdm


def subtokens(token):
    return [
        token[beginning:end]
        for beginning, end
        in combinations(range(len(token) + 1), r=2)
    ]


language = argv[1]
filename = "data/wikimedia/wiki.{}.txt".format(language)
with open(filename, "rt", encoding="utf-8") as f:
    num_lines = sum(1 for _ in tqdm(f, desc="Counting lines in {}".format(filename)))
    f.seek(0)
    lines = tqdm(f, total=num_lines, desc="Computing subterm frequency statistics")
    for line in lines:
        tokens = simple_preprocess(line, min_len=0, max_len=float("inf"))
        for token in tokens:
            for subtoken in subtokens(token):
                stdout.buffer.write("{}\n".format(subtoken).encode("utf-8"))
' $LANGUAGE | sort -u | python3 -c '
from io import TextIOWrapper
from sys import stdin


for line in TextIOWrapper(stdin.buffer, encoding="utf-8"):
    line = line.rstrip("\n")
    print(len(line))
' | sort | uniq -c | python3 -c '
from io import TextIOWrapper
import json
from sys import stdin


subterm_length_freqs = {}
for line in TextIOWrapper(stdin.buffer, encoding="utf-8"):
    frequency, length = line.strip().split(" ")
    length = int(length)
    frequency = int(frequency)
    subterm_length_freqs[length] = frequency

results = {
    "subterm_length_freqs": subterm_length_freqs,
}
print(json.dumps(results, sort_keys=True, indent=4))
' > data/wikimedia/wiki.$LANGUAGE.json
    fi

    trap '' EXIT

    python3 -c '
from io import TextIOWrapper
import json
from sys import argv, stdout


OPTIMUM_POINT_ESTIMATE = 4.91
SUBTERM_SIZE_FROM, SUBTERM_SIZE_TO = 1, 10


language = argv[1]
filename = "data/wikimedia/wiki.{}.json".format(language)
with open(filename, "rt", encoding="utf-8") as f:
    subterm_length_freqs = json.load(f)["subterm_length_freqs"]

total_subterms = sum(subterm_length_freqs.values())
best_i, best_j, best_coverage = None, None, float("inf")
for i in range(SUBTERM_SIZE_FROM, SUBTERM_SIZE_TO + 1):
    for j in range(i, SUBTERM_SIZE_TO + 1):
        coverage = 100.0 * sum(
            subterm_length_freqs[str(subterm_size)]
            for subterm_size in range(i, j + 1)
            if str(subterm_size) in subterm_length_freqs
        ) / total_subterms
        if abs(coverage - OPTIMUM_POINT_ESTIMATE) < abs(best_coverage - OPTIMUM_POINT_ESTIMATE):
            best_i, best_j, best_coverage = i, j, coverage
print(
    "Suggested subword sizes for {}: -minn {} -maxn {} ({:.2f}% n-gram coverage)".format(
        language,
        best_i,
        best_j,
        best_coverage,
    )
)
' $LANGUAGE
done
