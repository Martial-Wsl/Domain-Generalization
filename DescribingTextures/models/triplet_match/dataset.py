import random
import torch.utils.data as data
import torch
import torch.nn as nn

from models.layers.img_encoder import ResnetEncoder
from models.layers.sentence_encoder import make_encoder

from data_api.dataset_api import TextureDescriptionData
from models.layers.img_encoder import build_transforms
import math
import numpy as np
import random

class TripletTrainData(data.Dataset, TextureDescriptionData):

    def __init__(self, split='train', lang_input='phrase', neg_img=True, neg_lang=True, hard_samples=False, neg_margin=1.0, vec_dim=256,
                 img_feats=(2, 4), lang_encoder_method='bert', word_emb_dim=300, word_encoder=None, model = None):

        data.Dataset.__init__(self)
        TextureDescriptionData.__init__(self, phid_format='str')
        self.split = split
        self.lang_input = lang_input
        self.neg_img = neg_img
        self.neg_lang = neg_lang
        self.model = model

        self.hard_samples = hard_samples
        self.neg_margin = neg_margin
        self.vec_dim = vec_dim
        self.dist_fn = lambda v1, v2: (v1 - v2).pow(2).sum(dim=-1)

        self.resnet_encoder = ResnetEncoder(use_feats=img_feats)
        self.lang_embed = make_encoder(lang_encoder_method, word_emb_dim, word_encoder)
        self.img_encoder = nn.Sequential(self.resnet_encoder, nn.Linear(self.resnet_encoder.out_dim, vec_dim))
        self.lang_encoder = nn.Sequential(self.lang_embed, nn.Linear(self.lang_embed.out_dim, vec_dim))

        self.img_transform = build_transforms(is_train=False)

        self.pos_pairs = list()
        for img_i, img_name in enumerate(self.img_splits[self.split]):
            img_data = self.img_data_dict[img_name]
            if self.lang_input == 'phrase':
                self.pos_pairs += [(img_i, ph) for ph in img_data['phrase_ids']]
            elif self.lang_input == 'description':
                self.pos_pairs += [(img_i, desc_idx) for desc_idx in range(len(img_data['descriptions']))]
            else:
                raise NotImplementedError


        return

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, pair_i):
        if self.hard_samples == False:
            img_i, desc_i = self.pos_pairs[pair_i]
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)
            pos_lang = pos_img_data['descriptions'][desc_i]

            neg_lang = None
            if self.neg_lang:
                while True:
                    # tu prends une image au hasard
                    img_name = random.choice(self.img_splits[self.split])
                    if img_name == pos_img_data['image_name']:
                        # si c'est la meme que la pos tu recommence
                        continue
                    # tu prends une desc au hasard de cette image
                    neg_lang = random.choice(self.img_data_dict[img_name]['descriptions'])
                    if neg_lang not in pos_img_data['descriptions']:
                        # si elle n'est pas dans les desc de la pos image c'est bon
                        break

            neg_img = None
            if self.neg_img:
                while True:
                    neg_img_name = random.choice(self.img_splits[self.split])
                    neg_img_data = self.img_data_dict[neg_img_name]
                    if pos_lang not in neg_img_data['descriptions']:
                        break
                neg_img = self.load_img(neg_img_name)
                neg_img = self.img_transform(neg_img)

        else:  # hard samples
            img_i, desc_i = self.pos_pairs[pair_i]
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)
            pos_lang = pos_img_data['descriptions'][desc_i]

            with torch.no_grad():
                idxes = []
                for i in range(32):
                    idxes.append(random.randint(1, 3284))
                imgs = []

                img_rdy = pos_img_data['image']
                imgs.append(torch.unsqueeze(self.img_transform(img_rdy), dim=0))

                for img_k in idxes:
                    img_data_k = self.get_split_data(self.split, img_k, load_img=True)
                    img_rdy_k = img_data_k['image']
                    imgs.append(torch.unsqueeze(self.img_transform(img_rdy_k), dim=0))

                t_images = torch.Tensor(33, 3, 224, 224)
                torch.cat(imgs, out=t_images)

                idxt = []
                for i in range(32):
                    idxt.append(random.randint(1, 3284*9))
                descr = []

                descr.append(pos_img_data['descriptions'][desc_i])

                for desc_k in idxt:
                    current_img_data = self.get_split_data(self.split, math.floor(desc_k / 9), load_img=True)
                    descr.append(current_img_data['descriptions'][desc_k % 9])


                out_img = self.img_encoder(t_images)
                out_txt = self.model.lang_encoder(descr).cpu()

                f_pos_i = out_img[0]
                f_pos_p = out_txt[0]

                pos_i_neg_p = []
                pos_p_neg_i = []

                for l in range(1,33):
                    pos_i_neg_p.append(np.linalg.norm((f_pos_i - out_txt[l]).numpy(), ord=2))
                    pos_p_neg_i.append(np.linalg.norm((out_img[l] - f_pos_p).numpy(), ord=2))

                neg_p_idx = [x for _, x in sorted(zip(pos_i_neg_p, range(len(pos_i_neg_p))))][0]
                neg_i_idx = [x for _, x in sorted(zip(pos_p_neg_i, range(len(pos_p_neg_i))))][0]

                neg_img_data = self.get_split_data(self.split, idxes[neg_i_idx], load_img=True)
                neg_img = neg_img_data['image']
                neg_img = self.img_transform(neg_img)

                neg_lang_data = self.get_split_data(self.split, math.floor(idxt[neg_p_idx]/9) , load_img=True)
                neg_lang = neg_lang_data['descriptions'][idxt[neg_p_idx]%9]


        return pos_img, pos_lang, neg_img, neg_lang
