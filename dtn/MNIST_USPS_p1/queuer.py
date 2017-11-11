import zmq
import sys

def main():
    
    context = zmq.Context()
    
    queuerBe = context.socket(zmq.DEALER)
    queuerBe.bind('tcp://*:5570')
    
    queuerFe = context.socket(zmq.ROUTER)
    queuerFe.bind('tcp://*:5560')
    
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
