# Language modeling using pre-trained embeddings

This is a customized version of PyTorch's example 
[Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model)
The example trains the word embedding model from scratch, while we aim to load the pre-traind embedding model and evaluate it.

We train a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
The training script uses the Wikitext-2 dataset, with `train.txt`, `valid.txt` and `test.txt` pieces of texts provided.

File `all.txt` is initially used, to infer a dictionary of embeddings for every present word.

The texts are supposed to be pre-processed and tokens are retrieved just by splitting the raw text 
using `data.splitter = re.compile("[ \n\r\t\v\f\0]+")`

The trained model can then also be used by the generate script to generate new text.

### Usage:

```bash
python main.py --cuda --tied --emsize 100 --nhid 100 --batch_size 20 --language cs
--nlayers 1 --epochs 1 --fasttext_model {embeddings_folder}/wiki.cs.suggested.bin
```

#### Original usage documentation (training embeddings from scratch)

- should still work, but will no longer be maintained:

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --language LANGUAGE   language of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```
