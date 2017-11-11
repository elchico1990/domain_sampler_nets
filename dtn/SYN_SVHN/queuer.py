import zmq
import sys

def main():
    
    print 'Queueing!'
    
    context = zmq.Context()
    
    queuerBe = context.socket(zmq.DEALER)
    queuerFe = context.socket(zmq.ROUTER)
    
    
    
    if sys.argv[1] == 'adda':	    
	queuerBe.bind('tcp://*:5570')
	queuerFe.bind('tcp://*:5560')
    elif sys.argv[1] == 'adda_di':
	queuerBe.bind('tcp://*:5670')
	queuerFe.bind('tcp://*:5660')
    elif sys.argv[1] == 'fa':
	queuerBe.bind('tcp://*:5770')
	queuerFe.bind('tcp://*:5760')
    
    
    
    poll = zmq.Poller()
    poll.register(queuerFe, zmq.POLLIN)
    
    while True:
        sockets = dict(poll.poll())
        if queuerFe in sockets:
            ident, expDir = queuerFe.recv_multipart()
            queuerBe.send_string(expDir)
    
    
    frontend.close()
    backend.close()
    context.term()
    
if __name__ == '__main__':
    main()
