import os
import torch
from models import original_CALF, CALF, CALF_mTAN_no_patch, CALF_regular, CALF_2, CALF_3, CALF_4, CALF_5, CALF_6, \
    CALF_7, CALF_8, CALF_9, CALF_10, CALF_11, CALF_12, CALF_13, CALF_14, CALF_15, CALF_16, CALF_17, CALF_18, CALF_19, CALF_20, \
    CALF_21, CALF_23, CALF_24, CALF_25, CALF_26, CALF_27, CALF_29, CALF_31, CALF_32, CALF_33, CALF_34, CALF_35, CALF_36, CALF_37, \
    CALF_38, CALF_39, CALF_40, CALF_41, CALF_42, CALF_43, CALF_44, CALF_45, CALF_46, CALF_47, CALF_48, CALF_49, CALF_50, CALF_51, CALF_llama


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "CALF": CALF,
            "CALF_mTAN_no_patch": CALF_mTAN_no_patch,
            "CALF_regular": CALF_regular,
            "CALF_2": CALF_2,
            "CALF_3": CALF_3,
            "CALF_4": CALF_4,
            "CALF_5": CALF_5,
            "CALF_6": CALF_6,
            "CALF_7": CALF_7,
            "CALF_8": CALF_8,
            "CALF_9": CALF_9,
            "CALF_10": CALF_10,
            "CALF_11": CALF_11,
            "CALF_12": CALF_12,
            "CALF_13": CALF_13,
            "CALF_14": CALF_14,
            "CALF_15": CALF_15,
            "CALF_16": CALF_16,
            "CALF_17": CALF_17,
            "CALF_18": CALF_18,
            "CALF_19": CALF_19,
            "CALF_20": CALF_20,
            "CALF_21": CALF_21,
            "CALF_23": CALF_23,
            "CALF_24": CALF_24,
            "CALF_25": CALF_25,
            "CALF_26": CALF_26,
            "CALF_27": CALF_27,
            "CALF_29": CALF_29,
            "CALF_31": CALF_31,
            "CALF_32": CALF_32,
            "CALF_33": CALF_33,
            "CALF_34": CALF_34,
            "CALF_35": CALF_35,
            "CALF_36": CALF_36,
            "CALF_37": CALF_37,
            "CALF_38": CALF_38,
            "CALF_39": CALF_39,
            "CALF_40": CALF_40,
            "CALF_41": CALF_41,
            "CALF_42": CALF_42,
            "CALF_43": CALF_43,
            "CALF_44": CALF_44,
            "CALF_45": CALF_45,
            "CALF_46": CALF_46,
            "CALF_47": CALF_47,
            "CALF_48": CALF_48,
            "CALF_49": CALF_49,
            "CALF_50": CALF_50,
            "CALF_51": CALF_51,
            "CALF_llama": CALF_llama,
            "original_CALF": original_CALF
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
