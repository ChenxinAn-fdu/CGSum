########### configs ##########
import os


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        self.root_dir = None
        self.train_path = None
        self.vocab_path = None
        self.model_path = None
        self.log_root = "running_logs"
        self.decode_path = None
        self.test_result_path = 'pyrouge_root'
        if not os.path.exists(self.test_result_path):
            os.mkdir(self.test_result_path)

        self.baseline = False
        self.pointer_gen = True
        self.use_gpu = True
        self.coverage_at = 2
        self.is_coverage = False
        self.pooling = "join"
        self.vocab_size = 50000

        # GNN config
        self.neighbor_process = None
        self.graph_input_type = "abstract"
        self.GNN = "GAT"
        self.max_neighbor_num = 64
        self.min_neighbor_num = 0
        self.max_dec_steps = 200
        self.share_encoder = False
        self.num_layers = 2
        self.num_heads = 4
        self.num_out_heads = 4
        self.gnn_drop = 0.1
        self.sample_num = 8
        self.use_arg_max = False
        self.sample_trick = "none"
        self.neighbor_type = "cite"
        self.residual = True

        # inference
        self.f_t = "exp"
        self.constant_weight = 1.0
        self.l_s = 75
        self.beam_size = 5
        self.trigram_blocking = False

        # lstm config
        self.hidden_dim = 256
        self.emb_dim = 128
        self.batch_size = 16
        self.max_enc_steps = 500
        self.max_graph_enc_steps = 300
        self.min_dec_steps = 130

        self.lr = 0.15
        self.adagrad_init_acc = 0.1
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4
        self.max_grad_norm = 2.0  # 2.0

        self.cov_loss_wt = 1.0
        self.loss_wt = 1.0
        self.eps = 1e-12
        self.n_epochs = 5

        self.lr_coverage = 0.010  # 0.15
        self.n_hop = 1

        self.mode = "train"
        self.setting = "inductive"






