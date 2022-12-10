RANDOM_SEED=`date +%s`



python -u evolution.py --data-path /mnt/DP_disk6 --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-L.yaml \
--min-param-limits 1 --param-limits 100  --data-set CIFAR10 2>&1 |tee autoformer_search_${RANDOM_SEED}.log