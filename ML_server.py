import grpc
from concurrent import futures
from proto.ML_pb2 import *
from proto.ML_pb2_grpc import add_MLServiceServicer_to_server, MLServiceServicer
import socket, time, torch, gc

from data_manager import DatasetCreator
import config as cfg
from ML_models import init_model

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

IP = get_host_ip()

class MLService(MLServiceServicer):
    def __init__(self) -> None:
        super().__init__()
        self.data_creator = DatasetCreator(num_clients=100, dataset_name="CIFAR10", partition_method="uniform")
        self.testloader = self.data_creator.get_loader(is_test=True)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def SubmitMLTask(self, request, context):
        print("Accept a task")
        fn = request.fn
        node_id = request.node_id
        to_save = request.to_save
        trainloader = self.data_creator.get_loader(node_id, batch_sz=32, is_test=False, time_out=0, num_workers=0)
        self._local_train(fn, to_save, trainloader)
        print("Complete a task")
        return TaskResponse(result="Success")
    
    def SubmitTest(self, request, context):
        print("Test a model")
        fn = request.fn
        acc, loss = self._test(fn)
        print("Test finished")
        return TestResponse(accuracy=acc, loss=loss)
    
    def SubmitAggTask(self, request, context):
        print("Aggregation Task")
        fns = []
        for fn in request.fns:
            fns.append(fn)
        fns.sort()
        global_model_fn = request.fn
        local_models = []
        for fn in fns:
            local_models.append(torch.load(fn))
        for idx, local_model in enumerate(local_models):
            if idx == 0:
                continue
            for key in local_model.keys():
                local_models[0][key] = torch.add(local_models[0][key], local_model[key])

        for key in local_models[0].keys():
            local_models[0][key] = torch.div(local_models[0][key], len(local_models))

        torch.save(local_models[0], global_model_fn)
        return AggregateResponse(fn=global_model_fn)
        

    def _local_train(self, fn, to_save, trainloader, epoch=5, lr=1e-2, momentum=0.9, weight_decay=5e-4):
        model = init_model(cfg.model_name)
        model.load_state_dict(torch.load(fn))
        model.to(device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        step = 0
        model.train()
        while step < epoch:
            step += 1
            for (X, y) in trainloader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                output = model(X)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.to(torch.device('cpu'))
        torch.save(model.state_dict(), to_save)
        # del model
        # gc.collect()
        
    def _test(self, fn):
        model = init_model(cfg.model_name)
        model.load_state_dict(torch.load(fn))
        model = model.to(device=self.device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss().cuda()
        accuracy = loss = 0
        
        with torch.no_grad():
            for (X, y) in self.testloader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                output = model(X)
                loss += criterion(output, y).item()
                predicted = output.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()

            accuracy /= len(self.testloader.dataset)
            loss /= len(self.testloader)
            model.to(torch.device('cpu'))
            
        return accuracy, loss

    
def run():
    MAX_MESSAGE_LENGTH = 1024*1024*1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
               ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    add_MLServiceServicer_to_server(MLService(),server)
    server.add_insecure_port(IP+":9999")
    server.start()
    print("Start ML service at {}".format(IP+":9999"))
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run()