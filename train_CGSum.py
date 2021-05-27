import argparse
import os
import fitlog
import torch
from fastNLP import FitlogCallback, RandomSampler
from fastNLP import Trainer
from torch.optim import Adagrad

from data_util.config import Config
from data_util.dataloader import ScisummGraphLoader, PAD_TOKEN
from model.loss import SummLoss
from model.metric import FastRougeMetric
from model.model import CGSum
from model.callback import TrainCallback, LRDecayCallback
from data_util.logging import logger


def set_up_data():
    paths = {"train": os.path.join(config.train_path, "train.jsonl"),
             "dev": os.path.join(config.train_path, "val.jsonl")}

    datainfo, vocabs = ScisummGraphLoader(setting=args.setting).process(paths, config, args.load_vocab)
    logger.info('-' * 10 + "set up data done!" + '-' * 10)
    return datainfo, vocabs


def run_train():
    datainfo, vocabs = set_up_data()
    train_sampler = RandomSampler()
    criterion = SummLoss(config=config, padding_idx=vocabs.to_index(PAD_TOKEN))
    model = CGSum(config, vocab=vocabs)
    model.to(device)

    initial_lr = config.lr
    logger.info(f"learning rate = {initial_lr}")
    optimizer = Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr,
                        initial_accumulator_value=config.adagrad_init_acc)

    train_loader = datainfo.datasets["train"]
    valid_loader = datainfo.datasets["dev"]

    callbacks = [TrainCallback(config, patience=10), FitlogCallback(),
                 LRDecayCallback(optimizer.param_groups, steps=args.weight_decay_step)]
    trainer = Trainer(model=model, train_data=train_loader, optimizer=optimizer, loss=criterion,
                      batch_size=config.batch_size, check_code_level=-1, sampler=train_sampler,
                      n_epochs=config.n_epochs, print_every=100, dev_data=valid_loader, update_every=args.update_every,
                      metrics=FastRougeMetric(pred='prediction', art_oovs='article_oovs',
                                              abstract_sentences='abstract_sentences', config=config,
                                              vocab=datainfo.vocabs["vocab"]),
                      metric_key="rouge-l-f", validate_every=args.validate_every * args.update_every,
                      save_path=None,
                      callbacks=callbacks, use_tqdm=True)

    logger.info("-" * 5 + "start training" + "-" * 5)
    traininfo = trainer.train(load_best_model=True)

    logger.info('   | end of Train | time: {:5.2f}s | '.format(traininfo["seconds"]))
    logger.info('[INFO] best eval model in epoch %d and iter %d', traininfo["best_epoch"], traininfo["best_step"])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    # train config
    parser.add_argument('--visible_gpu', default=0, type=int, required=True)
    parser.add_argument("--load_vocab", default=True, type=str2bool)
    parser.add_argument("--max_dec_steps", default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--validate_every', type=int, default=2000)

    # graph config
    parser.add_argument('--graph_input_type', default="abstract")
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--GNN', default="GAT")
    parser.add_argument("--residual", default=True, type=str2bool)

    # graph or use [SEP] to encode nbr nodes
    parser.add_argument('--neighbor_process', default="graph", choices=["graph, sep"])
    parser.add_argument("--pooling", default="join", choices=["join", "max", " mean"])
    parser.add_argument("--gnn_drop", default=0.1, type=float)
    parser.add_argument("--max_neighbor_num", default=32, type=int,
                        help="set this value smaller if memory cost too much")
    parser.add_argument("--min_neighbor_num", default=0, type=int)
    parser.add_argument("--weight_decay_step", default=100, type=int)
    parser.add_argument("--share_encoder", default=False, type=str2bool)
    parser.add_argument("--max_graph_enc_steps", default=200, type=int)
    parser.add_argument('--max_enc_steps', default=500, type=int)
    parser.add_argument("--min_dec_steps", default=130, type=int,
                        help="we suggest setting min dec steps to 130 in SSN(inductive) and 140 in SSN(transductive)")
    parser.add_argument("--n_hop", default=1, type=int)
    # GNN
    parser.add_argument("--sample_num", default=8, type=int)
    parser.add_argument("--sample_trick", default="attention", type=str)
    parser.add_argument("--use_arg_max", default=True, type=str2bool)
    parser.add_argument("--constant_weight", default=0., type=float)
    parser.add_argument("--l_s", default=75, type=int)
    parser.add_argument("--neighbor_type", default="cite", type=str,
                        help="if set to both, all citing and reference papers will be added as neighbor")
    parser.add_argument("--f_t", default="exp")
    # lstm config
    parser.add_argument('--coverage_at', default=2, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--trigram_blocking', default=False, type=str2bool)

    # set paths
    parser.add_argument("--dataset_dir", default=None, help="path to training files")
    parser.add_argument("--vocab_file", default="vocab")
    parser.add_argument("--model_dir", default="save_models", help="path to save checkpoints")
    # we use  fitlog to save the best checkpoint during training
    parser.add_argument("--fitlog_dir", default="CGSum_fitlog", type=str)
    parser.add_argument("--setting", default="inductive", type=str, choices=["inductive", "transductive"])
    parser.add_argument('--mode', default="train")
    args = parser.parse_args()

    # write args to config file
    config = Config()
    config.train_path = args.dataset_dir if args.dataset_dir is not None else os.path.join("SSN", args.setting)
    config.vocab_path = os.path.join(config.train_path, args.vocab_file)
    config.model_path = args.model_dir

    # model parameters
    config.coverage_at = args.coverage_at
    config.trigram_blocking = args.trigram_blocking
    config.pooling = args.pooling
    config.gnn_drop = args.gnn_drop
    config.max_dec_steps = args.max_dec_steps
    config.batch_size = args.batch_size
    config.num_layers = args.num_layers
    config.GNN = args.GNN
    config.graph_input_type = args.graph_input_type
    config.neighbor_process = args.neighbor_process
    config.max_neighbor_num = args.max_neighbor_num
    config.min_neighbor_num = args.min_neighbor_num
    config.share_encoder = args.share_encoder
    config.n_hop = args.n_hop
    config.beam_size = args.beam_size

    # sample graph
    config.sample_num = args.sample_num
    config.use_arg_max = args.use_arg_max
    config.sample_trick = args.sample_trick
    config.neighbor_type = args.neighbor_type

    config.f_t = args.f_t
    config.constant_weight = args.constant_weight
    config.residual = args.residual
    config.max_enc_steps = args.max_enc_steps
    config.max_graph_enc_steps = args.max_graph_enc_steps
    config.min_dec_steps = args.min_dec_steps

    # mode
    config.mode = args.mode
    config.setting = args.setting

    # save model
    if not os.path.exists(config.model_path):
        if config.model_path.__contains__("/"):
            os.makedirs(config.model_path, 0o777)
        else:
            os.mkdir(config.model_path)

    # fitlog dir
    logger.info(f"set fitlog dir to {args.fitlog_dir}")
    if not os.path.exists(args.fitlog_dir):
        os.mkdir(args.fitlog_dir)
    fitlog.set_log_dir(args.fitlog_dir)
    fitlog.add_hyper(args)

    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    if args.visible_gpu != -1:
        config.use_gpu = True
        torch.cuda.set_device(args.visible_gpu)
        device = torch.device(args.visible_gpu)
    else:
        config.use_gpu = False

    mode = args.mode
    logger.info("------start mode train------")
    run_train()
