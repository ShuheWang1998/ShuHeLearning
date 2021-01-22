train_path = "/data/wangshuhe/learn/process_data/train.txt"
src_corpus = "/data/wangshuhe/learn/process_data/de.dict"
tar_corpus = "/data/wangshuhe/learn/process_data/en.dict"
dev_path = "/data/wangshuhe/learn/process_data/dev.txt"
test_path = "/data/wangshuhe/learn/process_data/test.txt"

cuda = True

# train
seed = 1
embed_size = 1000
hidden_size = 1000
window_size_d = 10
encoder_layer = 4
decoder_layers = 4
dropout_rate = 0.2
lr = 0.001
batch_size = 128
dev_batch_size = 128
# clip_grad
log_step = 1
valid_iter = 5
model_save_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/"
patience = 10
lr_decay = 0.5
max_epoch = 50


# test
checkpoint = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/Best_checkpoint.pth"
max_tar_length = 100