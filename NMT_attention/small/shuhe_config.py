train_path_src = "/data/wangshuhe/learn/process_data/LSTM/de_one.txt"
train_path_tar = "/data/wangshuhe/learn/process_data/LSTM/en_one.txt"
dev_path_src = "/data/wangshuhe/learn/process_data/LSTM/de_one.txt"
dev_path_tar = "/data/wangshuhe/learn/process_data/LSTM/en_one.txt"
test_path_src = "/data/wangshuhe/learn/process_data/LSTM/de_one.txt"
test_path_tar = "/data/wangshuhe/learn/process_data/LSTM/en_one.txt"
src_corpus = "/data/wangshuhe/learn/process_data/LSTM/de.txt"
tar_corpus = "/data/wangshuhe/learn/process_data/LSTM/en.txt"

cuda = True

# train
seed = 1
embed_size = 1000
hidden_size = 1000
window_size_d = 10
encoder_layer = 4
decoder_layers = 4
dropout_rate = 0.2
batch_size = 196
dev_batch_size = 196
clip_grad = 5
valid_iter = 20
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/small/result"
max_epoch = 10000
warm_up_step = 4000

# test
checkpoint = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/checkpoint.pth"
max_tar_length = 100
test_batch_size = 50
alpha = 0.7