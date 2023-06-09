# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.client_pb2 as client__pb2


class ClientServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.LocalTrain = channel.unary_unary(
                '/client.ClientService/LocalTrain',
                request_serializer=client__pb2.GlobalModel.SerializeToString,
                response_deserializer=client__pb2.LocalModel.FromString,
                )


class ClientServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def LocalTrain(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ClientServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'LocalTrain': grpc.unary_unary_rpc_method_handler(
                    servicer.LocalTrain,
                    request_deserializer=client__pb2.GlobalModel.FromString,
                    response_serializer=client__pb2.LocalModel.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'client.ClientService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ClientService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def LocalTrain(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/client.ClientService/LocalTrain',
            client__pb2.GlobalModel.SerializeToString,
            client__pb2.LocalModel.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
