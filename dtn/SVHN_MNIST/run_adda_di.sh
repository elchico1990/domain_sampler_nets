export LD_LIBRARY_PATH=/usr/local/cuda/lib64

tmux new-session -d -s 'train_adda_di' 'python main.py adda_di --mode=train_dsn'
tmux new-session -d -s 'queuer_adda_di' 'python queuer.py adda_di'
tmux new-session -d -s 'test_adda_di' 'python main.py adda_di test --mode=test'
