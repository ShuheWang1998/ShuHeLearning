train_path_src = "/data/wangshuhe/learn/process_data/LSTM/only_de_sen_train.txt"
train_path_tar = "/data/wangshuhe/learn/process_data/LSTM/only_en_sen_train.txt"
dev_path_src = "/data/wangshuhe/learn/process_data/LSTM/only_de_sen_dev.txt"
dev_path_tar = "/data/wangshuhe/learn/process_data/LSTM/only_en_sen_dev.txt"
test_path_src = "/data/wangshuhe/learn/process_data/LSTM/only_de_sen_test.txt"
test_path_tar = "/data/wangshuhe/learn/process_data/LSTM/only_en_sen_test.txt"
#corpus = "/data/wangshuhe/learn/process_data/shuhe/corpus.txt"
src_corpus = "/data/wangshuhe/learn/process_data/LSTM/de.txt"
tar_corpus = "/data/wangshuhe/learn/process_data/LSTM/en.txt"

embed_size = 512
# train
cuda = True
warm_up_step = 4000
train_batch_size = 96
max_epoch = 100000
valid_iter = 1
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
smoothing_eps = 0.1
# dev
dev_batch_size = 96
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/result/"
# test
# checkpoint
max_tar_length = 100
test_batch_size = 50
num_threads = 8
alpha = 0.7