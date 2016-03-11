import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

ADABOOST_PARAMS = {'n_estimators': 100, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'}
