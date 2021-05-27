import os
from collections import deque
import dgl
from fastNLP import Vocabulary
from fastNLP.io import JsonLoader
from data_util.logging import logger

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
SEP = '[SEP]'
INDEX = 0


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


def article2ids(intro, vocab, config, graph_inputs_wd=None):
    ids = []
    oovs = []
    unk_id = vocab.to_index(UNKNOWN_TOKEN)
    if not graph_inputs_wd:
        for w in intro:
            i = vocab.to_index(w)
            if i == unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(
                    len(vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
    else:
        for neighbor_input in graph_inputs_wd:
            for w in neighbor_input:
                i = vocab.to_index(w)
                if i == unk_id:  # If w is OOV
                    if w not in oovs:  # Add to list of OOVs
                        oovs.append(w)
                    oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                    ids.append(
                        len(vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                else:
                    ids.append(i)
    if len(oovs) == 0:
        oovs.append("N O N E")
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.to_index(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.to_index(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = len(vocab) + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def getting_full_info(enc_input_wd, graph_inputs_wd, abstract_wd, vocab, config):
    # Get ids of special tokens
    start_decoding = vocab.to_index(START_DECODING)
    stop_decoding = vocab.to_index(STOP_DECODING)

    # Process the graph input or input
    enc_input = [vocab.to_index(w) for w in enc_input_wd]
    nbr_inputs = []  # list of word ids; OOVs are represented by the id for UNK token
    for intro in graph_inputs_wd[:config.max_neighbor_num]:
        nbr_inputs.append([vocab.to_index(w) for w in intro])

    # Process the abstract
    abs_ids = [vocab.to_index(w) for w in abstract_wd]  # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    dec_input, target = get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    dec_len = len(dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id;
        enc_input_extend_vocab, article_oovs = article2ids(enc_input_wd, vocab, config, None)

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = abstract2ids(abstract_wd, vocab, article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)
    else:
        article_oovs = ["N O N E"]
        enc_input_extend_vocab = [-1]

    return enc_input, nbr_inputs, dec_input, target, dec_len, article_oovs, enc_input_extend_vocab


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.to_word(i)  # might be [UNK]
        except KeyError as e:  # w is OOV
            # assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            assert "N O N E" not in article_oovs, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - len(vocab)
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.to_index(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.to_index(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.to_index(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        # w is oov
        if vocab.to_index(w) == unk_token:
            # if article_oovs is None:  # baseline mode
            if "N O N E" in article_oovs:
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)

    out_str = ' '.join(new_words)
    return out_str


class ScisummGraphLoader(JsonLoader):
    """
    load summarization dataset SSN:

        text: list(str)，document
        summary: list(str), summary
        text_wd: list(list(str))，tokenized document
        summary_wd: list(list(str)), tokenized summary


    """

    def __init__(self, setting="inductive"):
        super(ScisummGraphLoader, self).__init__()
        self.setting = setting
        self.max_concat_len = 40
        self.max_concat_num = 5

    def _load(self, path):
        ds = super(ScisummGraphLoader, self)._load(path)
        return ds

    def process(self, paths, config, load_vocab_file=True):
        """
        :param paths: dict  path for each dataset
        :param load_vocab_file: bool  build vocab (False) or load vocab (True)
        :return: DataBundle
            datasets: dict  keys correspond to the paths dict
            vocabs: dict  key: vocab(if "train" in paths), domain(if domain=True), tag(if tag=True)
            embeddings: optional
        """

        vocab_size = config.vocab_size

        def _merge_abstracts(abstracts):
            merged = []
            for abstract in abstracts:
                merged.extend(abstract[:self.max_concat_len] + [SEP])
            if len(abstracts) == 0:
                assert merged == []
            return merged[:-1]

        def _pad_graph_inputs(graph_inputs):
            pad_text_wd = []
            max_len = config.max_graph_enc_steps

            for graph_input in graph_inputs:
                if len(graph_input) < max_len:
                    pad_num = max_len - len(graph_input)
                    graph_input.extend([PAD_TOKEN] * pad_num)
                else:
                    graph_input = graph_input[:max_len]
                pad_text_wd.append(graph_input)

            if len(pad_text_wd) == 0:
                pad_text_wd.append([PAD_TOKEN] * max_len)

            return pad_text_wd

        def _get_nbr_input_len(input_wd):
            enc_len = [min(len(text), config.max_graph_enc_steps) for text in input_wd]
            if len(enc_len) == 0:
                enc_len = [0]
            return enc_len

        def _pad_article(text_wd):
            token_num = len(text_wd)
            max_len = config.max_enc_steps
            if config.neighbor_process == "sep":
                max_len += self.max_concat_len * self.max_concat_num
            if token_num < max_len:
                padding = [PAD_TOKEN] * (max_len - token_num)
                article = text_wd + padding
            else:
                article = text_wd[:max_len]
            return article

        def _split_list(input_list):
            return [text.split() for text in input_list]

        def sent_tokenize(abstract):
            abs_list = abstract.split(".")
            return [(abst + ".") for abst in abs_list[:-1]]

        def _article_token_mask(text_wd):
            max_enc_len = config.max_enc_steps
            if config.neighbor_process == "sep":
                max_enc_len += self.max_concat_len * self.max_concat_num
            token_num = len(text_wd)
            if token_num < max_enc_len:
                mask = [1] * token_num + [0] * (max_enc_len - token_num)
            else:
                mask = [1] * max_enc_len
            return mask

        def generate_article_input(text, abstracts):
            if config.neighbor_process == "sep":
                text_wd = text.split()[:config.max_enc_steps]
                text_wd.append(SEP)
                abstracts_wd = _merge_abstracts(abstracts)
                return text_wd + abstracts_wd
            else:
                return text.split()

        def generate_graph_inputs(graph_struct):

            graph_inputs_ = [graph_strut_dict[pid][config.graph_input_type] for pid in graph_struct]
            return _split_list(graph_inputs_[1:])

        def generate_graph_structs(paper_id):
            sub_graph_dict = {}
            sub_graph_set = []

            n_hop = config.n_hop
            max_neighbor_num = config.max_neighbor_num
            k_nbrs = _k_hop_neighbor(paper_id, n_hop, max_neighbor_num)
            for sub_g in k_nbrs:
                sub_graph_set += sub_g

            for node in sub_graph_set:
                sub_graph_dict[node] = []

            for sub_g in k_nbrs:
                for centre_node in sub_g:
                    nbrs = graph_strut_dict[centre_node]['references']
                    c_nbrs = list(set(nbrs).intersection(sub_graph_set))
                    sub_graph_dict[centre_node].extend(c_nbrs)
                    for c_nbr in c_nbrs:
                        sub_graph_dict[c_nbr].append(centre_node)
            # in python 3.6, the first in subgraph dict is source paper
            return sub_graph_dict

        def _k_hop_neighbor(paper_id, n_hop, max_neighbor):
            sub_graph = [[] for _ in range(n_hop + 1)]
            level = 0
            visited = set()
            q = deque()
            q.append([paper_id, level])
            curr_node_num = 0
            while len(q) != 0:
                paper_first = q.popleft()
                paper_id_first, level_first = paper_first
                if level_first > n_hop:
                    return sub_graph
                sub_graph[level_first].append(paper_id_first)
                curr_node_num += 1
                if curr_node_num > max_neighbor:
                    return sub_graph
                visited.add(paper_id_first)
                for pid in graph_strut_dict[paper_id_first]["references"]:
                    if pid not in visited and pid in graph_strut_dict:
                        q.append([pid, level_first + 1])
                        visited.add(pid)

            return sub_graph

        def generate_dgl_graph(paper_id, graph_struct, nodes_num):
            g = dgl.DGLGraph()
            assert len(graph_struct) == nodes_num

            g.add_nodes(len(graph_struct))
            pid2idx = {}
            for index, key_node in enumerate(graph_struct):
                pid2idx[key_node] = index
            assert pid2idx[paper_id] == 0

            for index, key_node in enumerate(graph_struct):
                neighbor = [pid2idx[node] for node in graph_struct[key_node]]
                # add self loop
                neighbor.append(index)
                key_nodes = [index] * len(neighbor)
                g.add_edges(key_nodes, neighbor)
            return g

        train_ds = None
        dataInfo = self.load(paths)

        # pop nodes in train graph in inductive setting
        if config.mode == "test" and self.setting == "inductive":
            dataInfo.datasets.pop("train")

        graph_strut_dict = {}
        for key, ds in dataInfo.datasets.items():
            for ins in ds:
                graph_strut_dict[ins["paper_id"]] = ins

        logger.info(f"the input graph G_v has {len(graph_strut_dict)} nodes")

        for key, ds in dataInfo.datasets.items():
            # process summary
            ds.apply(lambda x: x['abstract'].split(), new_field_name='summary_wd')
            ds.apply(lambda x: sent_tokenize(x['abstract']), new_field_name='abstract_sentences')
            # generate graph

            ds.apply(lambda x: generate_graph_structs(x["paper_id"]), new_field_name="graph_struct")
            ds.apply(lambda x: generate_graph_inputs(x["graph_struct"]), new_field_name='graph_inputs_wd')

            ds.apply(lambda x: len(x["graph_inputs_wd"]) + 1, new_field_name="nodes_num")
            # pad input
            ds.apply(lambda x: generate_article_input(x['introduction'], x["graph_inputs_wd"]),
                     new_field_name='input_wd')
            ds.apply(lambda x: _article_token_mask(x["input_wd"]), new_field_name="enc_len_mask")
            ds.apply(lambda x: sum(x["enc_len_mask"]), new_field_name="enc_len")
            ds.apply(lambda x: _pad_article(x["input_wd"]), new_field_name="pad_input_wd")

            ds.apply(lambda x: _get_nbr_input_len(x["graph_inputs_wd"]), new_field_name="nbr_inputs_len")

            ds.apply(lambda x: _pad_graph_inputs(x["graph_inputs_wd"]), new_field_name="pad_graph_inputs_wd")
            if key == "train":
                train_ds = ds

        vocab_dict = {}
        if not load_vocab_file:
            logger.info("[INFO] Build new vocab from training dataset!")
            if train_ds is None:
                raise ValueError("Lack train file to build vocabulary!")

            vocabs = Vocabulary(max_size=config.vocab_size - 2, padding=PAD_TOKEN, unknown=UNKNOWN_TOKEN)
            vocabs.from_dataset(train_ds, field_name=["input_wd", "summary_wd"])
            vocabs.add_word(START_DECODING)
            vocabs.add_word(STOP_DECODING)
            vocab_dict["vocab"] = vocabs
            # save vocab
            with open(os.path.join(config.train_path, "vocab"), "w", encoding="utf8") as f:
                for w, idx in vocabs:
                    f.write(str(w) + "\t" + str(idx) + "\n")
            logger.info("build new vocab ends.. please reRun the code with load_vocab = True")
            exit(0)
        else:

            logger.info("[INFO] Load existing vocab from %s!" % config.vocab_path)
            word_list = []
            cnt = 3  # pad and unk
            if config.neighbor_process == "sep":
                cnt += 1

            with open(config.vocab_path, 'r', encoding='utf8') as vocab_f:
                for line in vocab_f:
                    pieces = line.split("\t")
                    word_list.append(pieces[0])
                    cnt += 1
                    if cnt > vocab_size:
                        break

            vocabs = Vocabulary(max_size=vocab_size, padding=PAD_TOKEN, unknown=UNKNOWN_TOKEN)
            vocabs.add_word_lst(word_list)
            vocabs.add(START_DECODING)
            vocabs.add(STOP_DECODING)
            if config.neighbor_process == "sep":
                vocabs.add(SEP)
            vocabs.build_vocab()
            vocab_dict["vocab"] = vocabs

        logger.info(f"vocab size = {len(vocabs)}")
        assert len(vocabs) == config.vocab_size
        dataInfo.set_vocab(vocabs, "vocab")

        for key, dataset in dataInfo.datasets.items():
            # do not process the training set in test mode
            if config.mode == "test" and key == "train":
                continue

            data_dict = {
                "enc_input": [],
                "nbr_inputs": [],
                "graph": [],
                "dec_input": [],
                "target": [],
                "dec_len": [],
                "article_oovs": [],
                "enc_input_extend_vocab": [],
            }
            logger.info(f"start construct the input of the model for {key} set, please wait...")
            for instance in dataset:
                graph_inputs = instance["pad_graph_inputs_wd"]
                abstract_sentences = instance["summary_wd"]
                enc_input = instance["pad_input_wd"]
                enc_input, nbr_inputs, dec_input, target, dec_len, article_oovs, enc_input_extend_vocab = \
                    getting_full_info(enc_input, graph_inputs, abstract_sentences, dataInfo.vocabs['vocab'], config)
                graph = generate_dgl_graph(instance["paper_id"], instance["graph_struct"], instance["nodes_num"])
                data_dict["graph"].append(graph)
                data_dict["enc_input"].append(enc_input)
                data_dict["nbr_inputs"].append(nbr_inputs)
                data_dict["dec_input"].append(dec_input)
                data_dict["target"].append(target)
                data_dict["dec_len"].append(dec_len)
                data_dict["article_oovs"].append(article_oovs)
                data_dict["enc_input_extend_vocab"].append(enc_input_extend_vocab)

            dataset.add_field("enc_input", data_dict["enc_input"])
            dataset.add_field("nbr_inputs", data_dict["nbr_inputs"])
            dataset.add_field("dec_input", data_dict["dec_input"])
            dataset.add_field("target", data_dict["target"])
            dataset.add_field("dec_len", data_dict["dec_len"])
            dataset.add_field("article_oovs", data_dict["article_oovs"])
            dataset.add_field("enc_input_extend_vocab", data_dict["enc_input_extend_vocab"])

            dataset.add_field("graph", data_dict["graph"])
            dataset.set_ignore_type('graph')  # without this line, there may be some errors
            dataset.set_input("graph")

            dataset.set_input("nbr_inputs_len", "nbr_inputs", "enc_len", "enc_input", "enc_len_mask",
                              "dec_input", "dec_len", "article_oovs", "nodes_num",
                              "enc_input_extend_vocab")
            dataset.set_target("target", "article_oovs", "abstract_sentences")

            dataset.delete_field('graph_inputs_wd')
            dataset.delete_field('pad_graph_inputs_wd')
            dataset.delete_field('input_wd')
            dataset.delete_field('pad_input_wd')
        logger.info("------load dataset over---------")
        return dataInfo, vocabs
