# FastText Subword Size Optimizer

 [![badge][]][colab]

 [badge]: https://colab.research.google.com/assets/colab-badge.svg
 [colab]: https://colab.research.google.com/github/MIR-MU/fasttext-optimizer/blob/master/correlate_language_distances.ipynb

Suggests subword sizes for fastText language models using character n-gram
frequency analysis.

## Suggesting Subword Sizes

To suggest subword sizes for one or more languages, use the
`suggest_subword_sizes.sh` tool:

``` sh
$ git clone https://github.com/MIR-MU/fasttext-optimizer.git --recurse-submodules
$ cd fasttext-optimizer
$ TMPDIR=/var/tmp ./suggest_subword_sizes.sh en de cs it

Suggested subword sizes for en: -minn 4 -maxn 5 (3.76% n-gram coverage)
Suggested subword sizes for de: -minn 6 -maxn 6 (4.19% n-gram coverage)
Suggested subword sizes for cs: -minn 1 -maxn 4 (3.28% n-gram coverage)
Suggested subword sizes for it: -minn 2 -maxn 5 (3.81% n-gram coverage)
```

To see how you can suggest subword sizes in Python, see also [our Python
tutorial][colab].

## Training FastText Models

To train one or more fastText models with the suggested subword sizes,
use the `train_fasttext_models.sh` tool:

```sh
$ git clone https://github.com/MIR-MU/fasttext-optimizer.git --recurse-submodules
$ cd fasttext-optimizer
$ TMPDIR=/var/tmp ./train_fasttext_models.sh cs de es fr
$ ls data/wikimedia/wiki.{cs,de,es,fr}.{default,suggested}.{bin,vec}

data/wikimedia/wiki.cs.default.bin    data/wikimedia/wiki.cs.default.vec
data/wikimedia/wiki.cs.suggested.bin  data/wikimedia/wiki.cs.suggested.vec
data/wikimedia/wiki.de.default.bin    data/wikimedia/wiki.de.default.vec
data/wikimedia/wiki.de.suggested.bin  data/wikimedia/wiki.de.suggested.vec
data/wikimedia/wiki.es.default.bin    data/wikimedia/wiki.es.default.vec
data/wikimedia/wiki.es.suggested.bin  data/wikimedia/wiki.es.suggested.vec
data/wikimedia/wiki.fr.default.bin    data/wikimedia/wiki.fr.default.vec
data/wikimedia/wiki.fr.suggested.bin  data/wikimedia/wiki.fr.suggested.vec
```

## Correlating Language Distances

To use suggested subword sizes as a measure of distance between languages and
to see how how this measure correlates with other language distance measures,
see [our Python tutorial][colab].
