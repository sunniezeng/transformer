# encoding: utf-8
import os

# raw_source_file = "../data/train/train.zh"
# raw_target_file = "../data/train/train.en"
# train_source_file = "../data/train/train1.zh"
# train_target_file = "../data/train/train1.en"
# test_source_file = "../data/train/test1.zh"
# test_target_file = "../data/train/test1.en"

source_file = "../data/train/news-commentary-v12.zh-en.zh"
target_file = "../data/train/news-commentary-v12.zh-en.en"
source_vocab_file = "../data/vocab/vocab.zh"
target_vocab_file = "../data/vocab/vocab.en"



checkpoint_dir = os.path.join(os.getcwd(), "saved_model", "transformer")
model_name = "transformer_v2"

min_count = 10
max_len = 5
batch_size = 1
num_units = 4
num_heads = 1
num_blocks = 6
num_epochs = 10000
dropout_rate = 0.1
is_training = True
sinusoid = False
