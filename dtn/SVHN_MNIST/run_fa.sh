export LD_LIBRARY_PATH=/usr/local/cuda/lib64

tmux new-session -d -s 'train_fa' 'python main.py fa --mode=train_dsn'
tmux new-session -d -s 'queuer_fa' 'python queuer.py fa'
tmux new-session -d -s 'test_fa' 'python main.py fa test --mode=test'


