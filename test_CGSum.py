import argparse
import glob
import os

import torch
from fastNLP import Tester
from data_util.logging import logger
from data_util.utils import write_eval_results
from model.metric import PyRougeMetric, FastRougeMetric
from model.model import CGSum
from data_util.dataloader import ScisummGraphLoader
from data_util.config import Config


def set_up_data(data_config):
    paths = {"train": os.path.join(data_config.train_path, args.train_file),
             "test": os.path.join(data_config.train_path, args.test_file)}

    datainfo, vocab = ScisummGraphLoader(setting=args.setting).process(paths, data_config, args.load_vocab)
    logger.info('-' * 10 + "set up data done!" + '-' * 10)
    return datainfo, vocab


def run_test():
    datainfo, vocab = set_up_data(config)
    model = CGSum(config, vocab)
    # load model state_dict
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if args.metrics == "pyrouge":
        metrics = PyRougeMetric(pred='prediction',
                                art_oovs='article_oovs',
                                abstract_sentences='abstract_sentences',
                                config=config,
                                vocab=datainfo.vocabs["vocab"])
    else:
        metrics = FastRougeMetric(pred='prediction', art_oovs='article_oovs',
                                  abstract_sentences='abstract_sentences', config=config,
                                  vocab=datainfo.vocabs["vocab"])

    tester = Tester(datainfo.datasets['test'], model=model, metrics=metrics,
                    batch_size=config.batch_size)
    eval_results = tester.test()
    if args.result_file_prefix is None:
        args.result_file_prefix = config.model_path.split("/")[-1]
    write_eval_results(args.result_dir, eval_results, args.result_file_prefix)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument('--visible_gpu', default=-1, type=int, required=True)
    parser.add_argument("--load_vocab", default=True, type=str2bool)
    parser.add_argument("--baseline", default=False, type=str2bool)
    parser.add_argument("--metrics", default="pyrouge", type=str, choices=["pyrouge", "fastrouge"])
    # inference
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument("--constant_weight", default=1.0, type=float,
                        help="the weight of rouge credit score, if set to 0 there will be no rouge credit")
    parser.add_argument("--trigram_blocking", default=True, type=str2bool)
    parser.add_argument("--min_dec_steps", default=130, type=int,
                        help="we suggest setting min dec steps to 130 in SSN(inductive) and 140 in SSN(transductive)")
    parser.add_argument("--max_dec_steps", default=200, type=int)
    parser.add_argument("--max_graph_enc_steps", default=300, type=int, help="input length of nbr nodes in the input graph")

    # path config
    parser.add_argument("--dataset_dir", default=None, help="dataset directory")
    parser.add_argument("--test_file", default="test.jsonl", help="test set file name")
    parser.add_argument("--train_file", default="train.jsonl", help="train set file name")
    parser.add_argument("--vocab_file", default="vocab")
    parser.add_argument("--model_dir", default="save_models",
                        help="the directory contains checkpoints after training")
    parser.add_argument("--model_name", default=None,
                        help="specifies the checkpoint, if it is None we will use the best cp in validation")
    parser.add_argument("--result_dir", default="results", help="path to the rouge score")
    parser.add_argument("--result_file_prefix", default=None, help="specifies the result file name")
    parser.add_argument("--decode_dir", default="decode_path", help="path to generated abstracts")
    parser.add_argument("--setting", default="inductive", choices=["transductive", "inductive"])
    args = parser.parse_args()

    config = Config()
    # load checkpoint
    if args.model_name is None:
        cpts = glob.glob(os.path.join(args.model_dir,
                                      "CGSum*"))
        cpts.sort(key=os.path.getmtime)
        # choice the last checkpoint by default
        cpt_file = cpts[-1]
    else:
        cpt_file = os.path.join(args.model_dir, args.model_name)
    logger.info(f"loading checkpoint from: {cpt_file}")

    checkpoint = torch.load(cpt_file)
    # load the config file
    config.__dict__ = checkpoint["config"]
    config.min_dec_steps = args.min_dec_steps
    config.max_dec_steps = args.max_dec_steps
    config.max_graph_enc_steps = args.max_graph_enc_steps
    # write args to config
    # paths config
    config.train_path = args.dataset_dir if args.dataset_dir is not None else os.path.join("SSN", args.setting)
    config.vocab_path = os.path.join(config.train_path, args.vocab_file)
    config.model_path = args.model_dir
    config.decode_path = args.decode_dir

    # add beam search config
    config.beam_size = args.beam_size
    config.trigram_blocking = args.trigram_blocking
    config.constant_weight = args.constant_weight

    # set mode to test
    config.mode = "test"

    # mkdir
    if not os.path.exists(config.decode_path):
        if config.decode_path.__contains__("/"):
            os.makedirs(config.decode_path, 0o777)
        else:
            os.mkdir(config.decode_path)

    if not os.path.exists(args.result_dir):
        if args.result_dir.__contains__("/"):
            os.makedirs(args.result_dir, 0o777)
        else:
            os.mkdir(args.result_dir)

    if args.visible_gpu != -1:
        config.use_gpu = True
        torch.cuda.set_device(args.visible_gpu)
        device = torch.device(args.visible_gpu)
    else:
        config.use_gpu = False

    logger.info("------start mode test-------")
    run_test()
