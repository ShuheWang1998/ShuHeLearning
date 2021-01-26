train_path = "/data/wangshuhe/learn/process_data/train.txt"
dev_path = "/data/wangshuhe/learn/process_data/dev.txt"
test_path = "/data/wangshuhe/learn/process_data/test.txt"
src_corpus = "/data/wangshuhe/learn/process_data/de.dict"
tar_corpus = "/data/wangshuhe/learn/process_data/en.dict"

embed_size = 512
# train
cuda = True
warm_up_step = 4000
train_batch_size = 180
max_epoch = 100000
valid_iter = 5
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
# dev
dev_batch_size = 180
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/result/"
# test
# checkpoint
max_tar_length = 100