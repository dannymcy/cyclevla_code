import pickle

import zmq


class ReasoningClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        server = "169.229.219.217"
        self.socket.connect(f"tcp://{server}:5623")

        self.result = None

    def request(self, args):
        self.socket.send(pickle.dumps(args))
        self.result = None

    def pull(self):
        if self.result is not None:
            return

        try:
            message = self.socket.recv(flags=zmq.NOBLOCK)
            self.result = pickle.loads(message)
        except zmq.Again:
            pass

    def done(self):
        self.pull()
        return self.result is not None

    def get_result(self):
        return self.result

    def join(self):
        if self.result is not None:
            return

        message = self.socket.recv()
        self.result = pickle.loads(message)
