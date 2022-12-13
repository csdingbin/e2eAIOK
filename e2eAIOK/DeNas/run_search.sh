RANDOM_SEED=`date +%s`
python -u search.py --domain cnn --conf ../../conf/denas/cv/e2eaiok_denas_cnn.conf 2>&1 | tee run_search_cnn_larger_flops_1000_epochs_${RANDOM_SEED}.log
# python -u search.py --domain vit --conf ../../conf/denas/cv/e2eaiok_denas_vit.conf 2>&1 | tee run_search_vit_1000_epochs_${RANDOM_SEED}.log