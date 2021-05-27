from __future__ import division

from fastNLP.core.metrics import MetricBase
from rouge import Rouge

from data_util.dataloader import STOP_DECODING
from data_util.dataloader import outputids2words
from data_util.logging import logger
from data_util.utils import pyrouge_score_all, pyrouge_score_all_multi


def remend_score(scores_all):
    remend_score = {}
    for key, value in scores_all.items():
        for subkey, subvalue in value.items():
            remend_score[key + "-" + subkey] = subvalue
    return remend_score


def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if "<" in s or ">" in s:
        print("-------html not safe sent:")
        print(s)
    return s


class RougeMetric(MetricBase):
    def __init__(self, pred=None, art_oovs=None, abstract_sentences=None, config=None, vocab=None):
        super().__init__()

        self.vocab = vocab
        self.config = config
        self._init_param_map(pred=pred, art_oovs=art_oovs, abstract_sentences=abstract_sentences)
        self.idx = 1
        self.prediction = []
        self.referece = []

    def evaluate(self, pred, art_oovs, abstract_sentences):
        """

        :param prediction: [batch, N]
        :param text: [batch, N]
        :param summary: [batch, N]
        :return:
        """
        batch_size = len(pred)

        for j in range(batch_size):
            # print(j,"----------------------",pred[j])
            output_ids = [int(id) for id in pred[j]]
            decoded_words = outputids2words(output_ids, self.vocab,
                                            (art_oovs[j] if self.config.pointer_gen else None))

            '''
            if batch_size == 1 and not isinstance(art_oovs[j], list):
                print("art oovs: ",art_oovs)
                decoded_words = outputids2words(output_ids, self.vocab, (art_oovs if self.config.pointer_gen else None))
            else:
                decoded_words = outputids2words(output_ids, self.vocab,
                                                (art_oovs[j] if self.config.pointer_gen else None))
            '''
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_sents = []
            while len(decoded_words) > 0:
                try:
                    fst_period_idx = decoded_words.index(".")
                except ValueError:
                    fst_period_idx = len(decoded_words)
                sent = decoded_words[:fst_period_idx + 1]
                decoded_words = decoded_words[fst_period_idx + 1:]
                decoded_sents.append(' '.join(sent))

            self.prediction.append("\n".join([make_html_safe(sent) for sent in decoded_sents]))
            self.referece.append("\n".join([make_html_safe(sent) for sent in abstract_sentences[j]]))
            '''
            if batch_size == 1 and not isinstance(abstract_sentences[j], list):
                #print("abstract sentences",abstract_sentences)
                self.referece.append(" ".join(abstract_sentences[0]))
            else:
                self.referece.append(" ".join(abstract_sentences[j]))
            '''

    def get_metric(self, reset=True):
        pass


class FastRougeMetric(RougeMetric):
    def __init__(self, pred=None, art_oovs=None, abstract_sentences=None, config=None, vocab=None):
        super().__init__(pred, art_oovs, abstract_sentences, config, vocab)

    def get_metric(self, reset=True):
        logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            logger.error("During testing, no hyps or refers is selected!")
            return
        rouge = Rouge()
        scores_all = rouge.get_scores(self.prediction, self.referece, avg=True)
        if reset:
            self.prediction = []
            self.referece = []
        logger.info(scores_all)
        scores_all = remend_score(scores_all)
        return scores_all


class PyRougeMetric(RougeMetric):
    def __init__(self, pred=None, art_oovs=None, abstract_sentences=None, config=None, vocab=None):
        super().__init__(pred, art_oovs, abstract_sentences, config, vocab)

    def get_metric(self, reset=True):
        logger.info("[INFO] Hyps and Refer number is %d, %d", len(self.prediction), len(self.referece))
        if len(self.prediction) == 0 or len(self.referece) == 0:
            logger.error("During testing, no hyps or refers is selected!")
            return
        if isinstance(self.referece[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = pyrouge_score_all_multi(self.prediction, self.referece, self.config)
        else:
            scores_all = pyrouge_score_all(self.prediction, self.referece, self.config)
        if reset:
            self.prediction = []
            self.referece = []
        logger.info(scores_all)
        return scores_all
