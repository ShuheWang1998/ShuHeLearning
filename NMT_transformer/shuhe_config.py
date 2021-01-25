train_path = "/data/wangshuhe/learn/process_data/one.txt"
dev_path = "/data/wangshuhe/learn/process_data/one.txt"
test_path = "/data/wangshuhe/learn/process_data/one.txt"
src_corpus = "/data/wangshuhe/learn/process_data/de_one.dict"
tar_corpus = "/data/wangshuhe/learn/process_data/en_one.dict"

embed_size = 512
# train
cuda = True
lr = 0.0001
train_batch_size = 128
max_epoch = 1000
valid_iter = 1
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.5
# dev
dev_batch_size = 128
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/result/"
patience = 10
lr_decay = 0.5
# test
# checkpoint
max_tar_length = 100