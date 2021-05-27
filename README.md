# CGSum 

code and dataset for AAAI 2021 paper: [Enhancing Scientific Papers Summarization with Citation Graph ](https://arxiv.org/pdf/2104.03057.pdf)


### DataSet SSN

The whole dataset and its corresponding citation relationship can be download through [this link](https://drive.google.com/file/d/1P5viA8hMm19n-Ia3k9wZyQTEloCk2gMJ/view?usp=sharing)

example of our dataset:

```
{
  "paper_id": "102498304", # unique id of this paper
  "title":"Weak Galerkin finite element method for Poisson’s ...", # title of this paper
  "abstract":"in this paper , the weak galerkin finite element method for second order eilliptc   problems employing polygonal or  ...", # human written abstract
  "text":[
  	["The weak galerkin finite element method using triangulated meshes was proposed by .."],
 	 ["Let @inlineform1 be a partition of the domain Ω consisting of polygons in two dimensional"], 
  	...
  ] # body text, 
  "section_names": ["Introduction", " Shape Regularity",  ...] # corresponding section names to sections
  "domain":"Mathematic", # class label
}
...
```
You can download our preprocessed dataset which can be directly loaded by `dataloader.py` via [SSN (inductive)](https://drive.google.com/file/d/1gxIniiwPHP53DEKdMMAJyHxM11Ib7dEd/view?usp=sharing
) and [SSN (transductive)](https://drive.google.com/file/d/14VSS1n1mo9irNhtDT1qvHA2JChKetaNh/view?usp=sharing). 
Note that we divide the dataset in two ways. The transductive division indicates that most neighbors of papers in test set are from the training set, but considering that in real cases, the test papers may from a new graph which has nothing to do with papers we used for training, thus we introduce SNN (inductive), by splitting the whole citation graph into three independent subgraphs – training, validation and test graphs.
Our preprocessed datasets are chunked to 500 words, for full document you can retrieve them from the whole dataset by `paper_id`

 
### requirements for running our code

- python 3.6+
- [PyTorch](https://pytorch.org/) 1.1+
- [DGL](http://dgl.ai) 0.4.3
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3

#### ROUGE Installation

we recommend using the following commands to install the ROUGE environment:

```shell
sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

You can refer to https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5 for RELEASE-1.5.5 and remember to build Wordnet 2.0 instead of 1.6 in RELEASE-1.5.5/data\

```shell
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

### Train and Test

Hyperparameters in the train.py/test.py script  has been set to default, we also provide the example to run our code in `train.sh` and `test.sh`.
you can train/test our model using the following command:

* **training**

```python
python train.py  --visible_gpu gpu_index  --model_dir  /path/to/save/checkpoints  --dataset_dir  path/to/dataset_directory --n_hop k
```

* **testing**

```python
python test.py   --visible_gpu gpu_index  --model_dir  /path/to/checkpoints  --model_name selected_checkpoint  --decode_dir /path/to/save/generated_abstract  --result_dir /path/to/save/ROUGE --n_hop 1
```

To test our model , remember to replace the pyrouge root set in `data_util/utils.py` to your own path.
you can also  download our trained model [Download](https://drive.google.com/file/d/13YoeNeQTLbsBIEuEIutACE55YI4wOVnm/view?usp=sharing)

our dataset is retrieved from [S2ORC](https://github.com/allenai/s2orc), the implementation of BertSum can refer to [PreSumm](https://github.com/nlpyang/PreSumm), thanks for their works.

