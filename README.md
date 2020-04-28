Suggests subword size hyperparameter for fastText language models using
character n-gram frequency analysis.

## Usage

### Suggesting subword sizes

``` sh
$ git submodule update --init --recursive
$ pip install -r requirements.txt
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
$ pip install -r requirements.txt
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

### Training language models

```sh
$ git submodule update --init --recursive
$ pip install -r requirements.txt
$ export TMPDIR=/var/tmp
$ ./train_language_models.sh cs de es fr
$ ls data/wmt13/wiki.{cs,de,es,fr}.lm.{default,suggested}.pt

data/wmt13/wiki.cs.lm.default.pt      data/wmt13/wiki.cs.lm.suggested.pt
data/wmt13/wiki.de.lm.default.pt      data/wmt13/wiki.de.lm.suggested.pt
data/wmt13/wiki.es.lm.default.pt      data/wmt13/wiki.es.lm.suggested.pt
data/wmt13/wiki.fr.lm.default.pt      data/wmt13/wiki.fr.lm.suggested.pt

$ for LOG_FILENAME in data/wmt13/wiki.{cs,de,es,fr}.lm.*.log.gz
> do
>     printf '%s:\t%s\n' $LOG_FILENAME "$(
>         gzip -d < "$LOG_FILENAME" | tail -n 7 |
>         sed -r -n '/test ppl/s/.*(test ppl\s*[0-9.]*).*/\1/p'
>     )"
> done

data/wmt13/wiki.cs.lm.default.log.gz:     test ppl  2005.62
data/wmt13/wiki.cs.lm.suggested.log.gz:   test ppl  1984.52

data/wmt13/wiki.de.lm.default.log.gz:     test ppl  1039.04
data/wmt13/wiki.de.lm.suggested.log.gz:   test ppl  1033.11

data/wmt13/wiki.es.lm.default.log.gz:     test ppl   275.69
data/wmt13/wiki.es.lm.suggested.log.gz:   test ppl   272.05

data/wmt13/wiki.fr.lm.default.log.gz:     test ppl   249.43
data/wmt13/wiki.fr.lm.suggested.log.gz:   test ppl   246.41

```
