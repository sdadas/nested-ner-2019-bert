#!/usr/bin/env python
from typing import Tuple, List
from collections import defaultdict

from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import RobertaProcessing

from reader.reader import Reader
import pickle

from config import config
from util.utils import save_dynamic_config


def batch_stat(batches: Tuple[List[List[List[int]]],
                              List[List[List[int]]],
                              List[List[List[int]]],
                              List[List[List[int]]],
                              List[List[List[Tuple[int, int, int]]]],
                              List[List[List[bool]]]]) -> None:
    all_num = 0
    start_num = 0
    end_num = 0
    for input_ids_batch, input_mask_batch, first_subtokens_batch, last_subtokens_batch, label_batch, mask_batch \
            in zip(*batches):
        for labels in label_batch:
            start_dic = defaultdict(list)
            end_dic = defaultdict(list)
            for ent in labels:
                start_dic[(ent[0], ent[2])].append(ent)
                end_dic[(ent[1], ent[2])].append(ent)
                all_num += 1
            for k, v in start_dic.items():
                if len(v) > 1:
                    start_num += len(v)
            for k, v in end_dic.items():
                if len(v) > 1:
                    end_num += len(v)

    print("All {}, start {}, end {}".format(all_num, start_num, end_num))


if __name__ == "__main__":
    tokenizer_dir = "tokenization/polish-roberta-base/"
    tokenizer = SentencePieceBPETokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")
    getattr(tokenizer, "_tokenizer").post_processor = RobertaProcessing(sep=("</s>", 2), cls=("<s>", 0))
    reader = Reader("polish", tokenizer, cls="<s>", sep="</s>", threshold=8)
    reader.read_all_data("./data/poleval/", "poleval.train", "poleval.dev", "poleval.test")

    # print reader.train_sents[0]
    train_batches, dev_batches, test_batches = reader.to_batch(config.batch_size)
    f = open(config.train_data_path, 'wb')
    pickle.dump(train_batches, f)
    f.close()

    f = open(config.dev_data_path, 'wb')
    pickle.dump(dev_batches, f)
    f.close()

    f = open(config.test_data_path, 'wb')
    pickle.dump(test_batches, f)
    f.close()

    batch_stat(train_batches)
    batch_stat(dev_batches)
    batch_stat(test_batches)

    misc_dict = save_dynamic_config(reader)
    misc_dict["bert_model"] = "./data/roberta_base_transformers/"
    f = open(config.config_data_path, 'wb')
    pickle.dump(misc_dict, f)
    f.close()

    print("Remember to scp word vectors as well")
