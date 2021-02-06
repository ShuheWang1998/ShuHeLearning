pre=/data/wangshuhe/learn/process_data/fairseq/small.de-en

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $pre/train \
    --validpref $pre/dev \
    --testpref $pre/test \
    --destdir /data/wangshuhe/learn/process_data/fairseq/preprocess/