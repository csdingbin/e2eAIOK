RANDOM_SEED=`date +%s`
python -u search.py --domain cnn --conf ../../conf/denas/cv/e2eaiok_denas_cnn.conf 2>&1 | tee run_search_de_score_${RANDOM_SEED}.log