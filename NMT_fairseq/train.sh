shuhedata=/data/wangshuhe/learn/process_data/fairseq/preprocess
savedir=/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_fairseq/result
CUDA_VISIBLE_DEVICES=1 fairseq-train $shuhedata \
    --max-sentences 128 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9\
    --lr 3e-4 --lr-scheduler inverse_sqrt \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch transformer \
    --save-interval-updates 1000 \
    --save-dir $savedir \
    --max-epoch 20 \
    --lr 3e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --keep-last-epochs 5