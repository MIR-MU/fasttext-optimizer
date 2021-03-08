Suggests subword size hyperparameter for fastText language models using
character n-gram frequency analysis.

## Usage

### Suggesting subword sizes

``` sh
$ git submodule update --init --recursive
$ export TMPDIR=/var/tmp
$ ./suggest_subword_sizes.sh en de cs it

Suggested subword sizes for en: -minn 4 -maxn 5 (3.76% n-gram coverage)
Suggested subword sizes for de: -minn 6 -maxn 6 (4.19% n-gram coverage)
Suggested subword sizes for cs: -minn 1 -maxn 4 (3.28% n-gram coverage)
Suggested subword sizes for it: -minn 2 -maxn 5 (3.81% n-gram coverage)
```

### Training fastText models

```sh
$ git submodule update --init --recursive
$ export TMPDIR=/var/tmp
$ ./train_fasttext_models.sh cs de es fr
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
