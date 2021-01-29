train_path = "/data/wangshuhe/learn/process_data/tiny.txt"
dev_path = "/data/wangshuhe/learn/process_data/tiny.txt"
test_path = "/data/wangshuhe/learn/process_data/tiny.txt"
src_corpus = "/data/wangshuhe/learn/process_data/de.dict"
tar_corpus = "/data/wangshuhe/learn/process_data/en.dict"

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
valid_iter = 5
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/small/result/"
max_epoch = 10000
patience = 10

# test
checkpoint = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/checkpoint.pth"
max_tar_length = 100