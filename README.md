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


codes, checkpoints, and generated abstracts of test set will be released soon.
 
our dataset is retrieved from [S2ORC](https://github.com/allenai/s2orc), the implementation of BertSum can refer to [PreSumm](https://github.com/nlpyang/PreSumm), thanks for their works.
