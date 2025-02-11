import csv
import random
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class HotelData(Dataset):

    def __init__(self, data_dir, aspect, mode, word2idx, max_length=256, balance=False):
        super(HotelData, self).__init__()
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.inputs = []
        self.masks = []
        self.labels = []
        self.path = os.path.join(data_dir, 'hotel_{}.{}'.format(self.num_to_aspect[aspect], mode))
        examples = self._create_examples(self._read_csv(self.path), mode, balance=balance)
        self._convert_examples_to_arrays(examples, max_length, word2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.inputs[item], self.masks[item], self.labels[item]

    def _read_csv(self, file_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(file_path, "rt", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, mode, balance=False):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = int(line[1])
            text = line[2]
            examples.append({'text': text, "label": label})

        print('Dataset: Hotel Review')
        print('{} samples has {}'.format(mode, len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (mode, len(pos_examples), len(neg_examples)))

        if balance:

            random.seed(20226666)

            print('Make the Training dataset class balanced.')

            min_examples = min(len(pos_examples), len(neg_examples))

            if len(pos_examples) > min_examples:
                pos_examples = random.sample(pos_examples, min_examples)

            if len(neg_examples) > min_examples:
                neg_examples = random.sample(neg_examples, min_examples)

            assert (len(pos_examples) == len(neg_examples))
            examples = pos_examples + neg_examples
            print(
                'After balance training data: %d positive examples, %d negative examples.'
                % (len(pos_examples), len(neg_examples)))
        return examples

    def _convert_single_text(self, text, max_length, word2idx):
        """
        Converts a single text into a list of ids with mask.
        """
        input_ids = []

        text_ = text.strip().split(" ")

        if len(text_) > max_length:
            text_ = text_[0:max_length]

        for word in text_:
            word = word.strip()
            try:
                input_ids.append(word2idx[word])
            except:
                # if the word is not exist in word2idx, use <unknown> token
                input_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # zero-pad up to the max_seq_length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(self, examples, max_length, word2idx):
        """
        Convert a set of train/dev examples numpy arrays.
        Outputs:
            data -- (num_examples, max_seq_length).
            masks -- (num_examples, max_seq_length).
            labels -- (num_examples, num_classes) in a one-hot format.
        """

        data = []
        labels = []
        masks = []
        for example in examples:
            input_ids, input_mask = self._convert_single_text(example["text"],
                                                              max_length, word2idx)

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))


class HotelAnnotation(Dataset):

    def __init__(self, data_dir, aspect, word2idx, max_length=256):
        super(HotelAnnotation, self).__init__()
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.input_ids = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'hotel_{}.train'.format(self.num_to_aspect[aspect]))),
            word2idx,
            max_length)

    def __getitem__(self, i):
        return self.input_ids[i], self.masks[i], self.labels[i], self.rationales[i]

    def __len__(self):
        return len(self.labels)

    def _read_tsv(self, annotation_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(annotation_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, word2idx, max_length):
        data = []
        labels = []
        masks = []
        rationales = []

        print('Dataset: Hotel Review')

        for i, line in enumerate(lines):
            if i == 0:
                continue
            text_ = line[2].split(" ")
            label_ = int(line[1])
            rationale = [int(x) for x in line[3].split(" ")]
            # process the text
            input_ids = []
            if len(text_) > max_length:
                text_ = text_[0:max_length]

            for word in text_:
                word = word.strip()
                try:
                    input_ids.append(word2idx[word])
                except:
                    # word is not exist in word2idx, use <unknown> token
                    input_ids.append(0)
            # process mask
            # The mask has 1 for real word and 0 for padding tokens.
            input_mask = [1] * len(input_ids)

            # zero-pad up to the max_seq_length.
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)

            assert (len(input_ids) == max_length)
            assert (len(input_mask) == max_length)

            # construct rationale
            binary_rationale = [0] * len(input_ids)
            for k in range(len(binary_rationale)):
                # print(k)
                if k < len(rationale):
                    binary_rationale[k] = rationale[k]

            data.append(input_ids)
            labels.append(label_)
            masks.append(input_mask)
            rationales.append(binary_rationale)

        self.input_ids = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))
        self.rationales = torch.from_numpy(np.array(rationales))


        sparsity = torch.sum(self.rationales) / torch.sum(self.masks)
        print('The sparsity is {}'.format(sparsity))


        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' % (pos, neg))

class ToyHotelData(Dataset):

    def __init__(self, data_dir, aspect, word2idx, max_length=256):
        super(ToyHotelData, self).__init__()
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.input_ids = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'hotel_{}.train'.format(self.num_to_aspect[aspect]))),
            word2idx,
            max_length)

    def __getitem__(self, i):
        return self.input_ids[i], self.masks[i], self.labels[i], self.rationales[i]

    def __len__(self):
        return len(self.labels)

    def _read_tsv(self, annotation_path, quotechar=None):
        """Reads a tab separated value file."""
        with open(annotation_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, word2idx, max_length):
        data = []
        labels = []
        masks = []
        rationales = []

        print('Dataset: Hotel Review')

        for i, line in enumerate(lines):
            if i == 0:
                continue
            text_ = line[2].split(" ")
            label_ = int(line[1])
            rationale = [int(x) for x in line[3].split(" ")]
            # process the text
            input_ids = []
            if len(text_) > max_length:
                text_ = text_[0:max_length]

            for word in text_:
                word = word.strip()
                try:
                    input_ids.append(word2idx[word])
                except:
                    # word is not exist in word2idx, use <unknown> token
                    input_ids.append(0)
            # process mask
            # The mask has 1 for real word and 0 for padding tokens.
            input_mask = [1] * len(input_ids)

            # zero-pad up to the max_seq_length.
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)

            assert (len(input_ids) == max_length)
            assert (len(input_mask) == max_length)

            # construct rationale
            binary_rationale = [0] * len(input_ids)
            for k in range(len(binary_rationale)):
                # print(k)
                if k < len(rationale):
                    binary_rationale[k] = rationale[k]

            data.append(input_ids)
            labels.append(label_)
            masks.append(input_mask)
            rationales.append(binary_rationale)

        self.input_ids = torch.from_numpy(np.array(data))
        self.masks = torch.from_numpy(np.array(masks))
        self.labels = torch.from_numpy(np.array(labels))
        self.rationales = torch.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = torch.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' %
              (pos, neg))

def split_dataset(all_data):
    pos_samples = []
    neg_samples = []
    for i,item in enumerate(all_data):
        input_id, mask, label, rationale = item
        if(label==1):
            pos_samples.append(item)
        else:
            neg_samples.append(item)

    train_set,dev_set,test_set = [],[],[]

    for i,item in enumerate(pos_samples):
        if (i%10==9):
            dev_set.append(item)
        elif(i%10==8):
            test_set.append(item)
        else:
            train_set.append(item)
    for i,item in enumerate(neg_samples):
        if (i%10==1):
            dev_set.append(item)
        elif(i%10==2):
            test_set.append(item)
        else:
            train_set.append(item)
    return train_set,dev_set,test_set

# from embedding import get_glove_embedding
# if __name__=="__main__":
#     embedding_dir,embedding_name = './data/hotel/embeddings','glove.6B.100d.txt'
#     pretrained_embedding, word2idx = get_glove_embedding(os.path.join(embedding_dir, embedding_name))
#     annotation_path = './data/hotel/annotations'
#     aspect = 0

#     all_data = ToyHotelData(annotation_path, aspect, word2idx)
#     # input_ids, masks, labels, rationales

#     train_set,dev_set,test_set = split_dataset(all_data)
#     print(len(train_set),len(dev_set),len(test_set))



from embedding import get_glove_embedding
if __name__=="__main__":
    embedding_dir,embedding_name = './data/hotel/embeddings','glove.6B.100d.txt'
    pretrained_embedding, word2idx = get_glove_embedding(os.path.join(embedding_dir, embedding_name))
    aspect = 1
    annotation_path = './data/hotel/annotations'
    all_data = HotelAnnotation(annotation_path, aspect, word2idx)
    # input_ids, masks, labels, rationales


'''
HotelData as0: The sparsity is 0.08483771979808807
HotelData as1: The sparsity is 0.11460386216640472
HotelData as2: The sparsity is 0.11460386216640472
'''