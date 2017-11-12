export LD_LIBRARY_PATH=/usr/local/cuda/lib64

tmux new-session -d -s 'train_adda' 'python main.py adda --mode=train_dsn'
tmux new-session -d -s 'queuer_adda' 'python queuer.py adda'
tmux new-session -d -s 'test_adda' 'python main.py adda test --mode=test'


