import os
from typing import Optional, List, Tuple
import pdb
from collections import namedtuple, defaultdict

from tokenizers import BertWordPieceTokenizer
from tokenizers.implementations import BaseTokenizer

from util.utils import Alphabet


SentInst = namedtuple('SentInst', 'tokens chars entities')



class Reader:
    def __init__(self, bert_model: str, tokenizer: BaseTokenizer=None, cls: str="[CLS]", sep: str="[SEP]", threshold=6):

        self.tokenizer: BaseTokenizer = tokenizer
        self.cls = cls
        self.sep = sep
        if self.tokenizer is None:
            vocab_path: str = "tokenization/" + bert_model + ".txt"
            self.tokenizer = BertWordPieceTokenizer(vocab_path, lowercase="-cased" not in bert_model)

        self.threshold = threshold
        self.subword_alphabet: Optional[Alphabet] = None
        self.label_alphabet: Optional[Alphabet] = None

        self.train: Optional[List[SentInst]] = None
        self.dev: Optional[List[SentInst]] = None
        self.test: Optional[List[SentInst]] = None

    def _read_file(self, filename: str, mode: str = 'train') -> List[SentInst]:
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":  # last few blank lines
                    break

                raw_tokens = line.split(' ')
                tokens = raw_tokens
                chars = [list(t) for t in raw_tokens]

                entities = next(f).strip()
                if entities == "":  # no entities
                    sent_inst = SentInst(tokens, chars, [])
                else:
                    entity_list = []
                    entities = entities.split("|")
                    for item in entities:
                        pointers, label = item.split()
                        pointers = pointers.split(",")
                        if int(pointers[1]) > len(tokens):
                            pdb.set_trace()
                        span_len = int(pointers[1]) - int(pointers[0])
                        if span_len < 0:
                            print("Warning! span_len < 0")
                            continue
                        if span_len > max_len:
                            max_len = span_len
                        if span_len > self.threshold:
                            num_thresh += 1

                        new_entity = (int(pointers[0]), int(pointers[1]), label)
                        # may be duplicate entities in some datasets
                        if (mode == 'train' and new_entity not in entity_list) or (mode != 'train'):
                            entity_list.append(new_entity)

                    # assert len(entity_list) == len(set(entity_list)) # check duplicate
                    sent_inst = SentInst(tokens, chars, entity_list)
                assert next(f).strip() == ""  # separating line

                sent_list.append(sent_inst)
        print("Max length: {}".format(max_len))
        print("Threshold {}: {}".format(self.threshold, num_thresh))
        return sent_list

    def _gen_dic(self) -> None:
        label_set = set()

        for sent_list in [self.train, self.dev, self.test]:
            num_mention = 0
            for sentInst in sent_list:
                for entity in sentInst.entities:
                    label_set.add(entity[2])
                num_mention += len(sentInst.entities)
            print("# mentions: {}".format(num_mention))

        vocab = [self.tokenizer.id_to_token(idx) for idx in range(self.tokenizer.get_vocab_size())]
        self.subword_alphabet = Alphabet(vocab, 0)
        self.label_alphabet = Alphabet(label_set, 0)

    @staticmethod
    def _pad_batches(input_ids_batches: List[List[List[int]]],
                     first_subtokens_batches: List[List[List[int]]]) \
            -> Tuple[List[List[List[int]]],
                     List[List[List[int]]],
                     List[List[List[bool]]]]:

        padded_input_ids_batches = []
        input_mask_batches = []
        mask_batches = []

        all_batches = list(zip(input_ids_batches, first_subtokens_batches))
        for input_ids_batch, first_subtokens_batch in all_batches:

            batch_len = len(input_ids_batch)
            max_subtokens_num = max([len(input_ids) for input_ids in input_ids_batch])
            max_sent_len = max([len(first_subtokens) for first_subtokens in first_subtokens_batch])

            padded_input_ids_batch = []
            input_mask_batch = []
            mask_batch = []

            for i in range(batch_len):

                subtokens_num = len(input_ids_batch[i])
                sent_len = len(first_subtokens_batch[i])

                padded_subtoken_vec = input_ids_batch[i].copy()
                padded_subtoken_vec.extend([0] * (max_subtokens_num - subtokens_num))
                input_mask = [1] * subtokens_num + [0] * (max_subtokens_num - subtokens_num)
                mask = [True] * sent_len + [False] * (max_sent_len - sent_len)

                padded_input_ids_batch.append(padded_subtoken_vec)
                input_mask_batch.append(input_mask)
                mask_batch.append(mask)

            padded_input_ids_batches.append(padded_input_ids_batch)
            input_mask_batches.append(input_mask_batch)
            mask_batches.append(mask_batch)

        return padded_input_ids_batches, input_mask_batches, mask_batches

    def to_batch(self, batch_size: int) -> Tuple:
        ret_list = []

        for sent_list in [self.train, self.dev, self.test]:
            subtoken_dic_dic = defaultdict(lambda: defaultdict(list))
            first_subtoken_dic_dic = defaultdict(lambda: defaultdict(list))
            last_subtoken_dic_dic = defaultdict(lambda: defaultdict(list))
            label_dic_dic = defaultdict(lambda: defaultdict(list))

            this_input_ids_batches = []
            this_first_subtokens_batches = []
            this_last_subtokens_batches = []
            this_label_batches = []

            for sentInst in sent_list:

                subtoken_vec = []
                first_subtoken_vec = []
                last_subtoken_vec = []
                subtoken_vec.append(self.tokenizer.token_to_id(self.cls))
                for t in sentInst.tokens:
                    encoding = self.tokenizer.encode(t)
                    ids = [v for v, mask in zip(encoding.ids, encoding.special_tokens_mask) if mask == 0]
                    first_subtoken_vec.append(len(subtoken_vec))
                    subtoken_vec.extend(ids)
                    last_subtoken_vec.append(len(subtoken_vec))
                subtoken_vec.append(self.tokenizer.token_to_id(self.sep))

                label_list = [(u[0], u[1], self.label_alphabet.get_index(u[2])) for u in sentInst.entities]

                subtoken_dic_dic[len(sentInst.tokens)][len(subtoken_vec)].append(subtoken_vec)
                first_subtoken_dic_dic[len(sentInst.tokens)][len(subtoken_vec)].append(first_subtoken_vec)
                last_subtoken_dic_dic[len(sentInst.tokens)][len(subtoken_vec)].append(last_subtoken_vec)
                label_dic_dic[len(sentInst.tokens)][len(subtoken_vec)].append(label_list)

            input_ids_batches = []
            first_subtokens_batches = []
            last_subtokens_batches = []
            label_batches = []
            for length1 in sorted(subtoken_dic_dic.keys(), reverse=True):
                for length2 in sorted(subtoken_dic_dic[length1].keys(), reverse=True):
                    input_ids_batches.extend(subtoken_dic_dic[length1][length2])
                    first_subtokens_batches.extend(first_subtoken_dic_dic[length1][length2])
                    last_subtokens_batches.extend(last_subtoken_dic_dic[length1][length2])
                    label_batches.extend(label_dic_dic[length1][length2])

            [this_input_ids_batches.append(input_ids_batches[i:i + batch_size])
             for i in range(0, len(input_ids_batches), batch_size)]
            [this_first_subtokens_batches.append(first_subtokens_batches[i:i + batch_size])
             for i in range(0, len(first_subtokens_batches), batch_size)]
            [this_last_subtokens_batches.append(last_subtokens_batches[i:i + batch_size])
             for i in range(0, len(last_subtokens_batches), batch_size)]
            [this_label_batches.append(label_batches[i:i + batch_size])
             for i in range(0, len(label_batches), batch_size)]

            this_input_ids_batches, this_input_mask_batches, this_mask_batches \
                = self._pad_batches(this_input_ids_batches, this_first_subtokens_batches)

            ret_list.append((this_input_ids_batches,
                             this_input_mask_batches,
                             this_first_subtokens_batches,
                             this_last_subtokens_batches,
                             this_label_batches,
                             this_mask_batches))

        return tuple(ret_list)

    def read_all_data(self, file_path: str, train_file: str, dev_file: str, test_file: str) -> None:
        self.train = self._read_file(file_path + train_file)
        self.dev = self._read_file(file_path + dev_file, mode='dev')
        self.test = self._read_file(file_path + test_file, mode='test')
        self._gen_dic()

    def debug_single_sample(self,
                            subtoken: List[int],
                            label_list: List[Tuple[int, int, int]]) -> None:
        print(" ".join([self.subword_alphabet.get_instance(t) for t in subtoken]))
        for label in label_list:
            print(label[0], label[1], self.label_alphabet.get_instance(label[2]))
