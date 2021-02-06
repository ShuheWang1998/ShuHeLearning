test_path=/data/wangshuhe/learn/process_data/fairseq/preprocess
model=/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_fairseq/result/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=1 fairseq-generate $test_path \
  --path $model \
  --beam 100 \
  --batch-size 5 \
  --remove-bpe \
  --gen-subset 'test' \
  --lenpen 0.6 \
  --quiet