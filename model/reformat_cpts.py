import torch
import os
from data_util.config import Config


def reformat_cpts():
    model_dir = "save_models"
    cpts = ["CGSum_inductive_1hopNbrs.pt", "CGSum_inductive_2hopNbrs.pt", "CGSum_transductive_1hopNbrs.pt",
            "CGSum_transductive_2hopNbrs.pt"]
    for cpt in cpts:
        cpt_file = os.path.join(model_dir, cpt)
        check_point = torch.load(cpt_file)
        check_point.pop("vocab")
        old_config_dict = check_point["config"]
        config_dict = Config().__dict__
        for key in config_dict:
            if key not in old_config_dict:
                print(key)
                continue
            config_dict[key] = old_config_dict[key]
        if "inductive" in cpt_file:
            config_dict["setting"] = "inductive"
            config_dict["min_dec_steps"] = 125
        else:
            config_dict["setting"] = "transductive"
            config_dict["min_dec_steps"] = 140
        config_dict["baseline"] = False
        config_dict["l_s"] = 75
        config_dict["f_t"] = "exp"
        check_point["config"] = config_dict
        new_cpt_file = os.path.join(model_dir, "new_cpts", cpt)
        torch.save(check_point, new_cpt_file)

if __name__ == '__main__':
    reformat_cpts()
