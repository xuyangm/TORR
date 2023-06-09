# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.node_pb2 as node__pb2


class NodeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetChunk = channel.unary_unary(
                '/node.NodeService/GetChunk',
                request_serializer=node__pb2.GetChunkRequest.SerializeToString,
                response_deserializer=node__pb2.GetChunkResponse.FromString,
                )
        self.Verify = channel.unary_unary(
                '/node.NodeService/Verify',
                request_serializer=node__pb2.VerifyRequest.SerializeToString,
                response_deserializer=node__pb2.VerifyResponse.FromString,
                )
        self.SaveChunk = channel.unary_unary(
                '/node.NodeService/SaveChunk',
                request_serializer=node__pb2.SaveChunkRequest.SerializeToString,
                response_deserializer=node__pb2.SaveChunkResponse.FromString,
                )
        self.DeliverBlock = channel.unary_unary(
                '/node.NodeService/DeliverBlock',
                request_serializer=node__pb2.grpcBlock.SerializeToString,
                response_deserializer=node__pb2.BlockResponse.FromString,
                )
        self.DeliverModel = channel.unary_unary(
                '/node.NodeService/DeliverModel',
                request_serializer=node__pb2.grpcModel.SerializeToString,
                response_deserializer=node__pb2.ModelResponse.FromString,
                )


class NodeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetChunk(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Verify(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SaveChunk(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeliverBlock(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeliverModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NodeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetChunk': grpc.unary_unary_rpc_method_handler(
                    servicer.GetChunk,
                    request_deserializer=node__pb2.GetChunkRequest.FromString,
                    response_serializer=node__pb2.GetChunkResponse.SerializeToString,
            ),
            'Verify': grpc.unary_unary_rpc_method_handler(
                    servicer.Verify,
                    request_deserializer=node__pb2.VerifyRequest.FromString,
                    response_serializer=node__pb2.VerifyResponse.SerializeToString,
            ),
            'SaveChunk': grpc.unary_unary_rpc_method_handler(
                    servicer.SaveChunk,
                    request_deserializer=node__pb2.SaveChunkRequest.FromString,
                    response_serializer=node__pb2.SaveChunkResponse.SerializeToString,
            ),
            'DeliverBlock': grpc.unary_unary_rpc_method_handler(
                    servicer.DeliverBlock,
                    request_deserializer=node__pb2.grpcBlock.FromString,
                    response_serializer=node__pb2.BlockResponse.SerializeToString,
            ),
            'DeliverModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeliverModel,
                    request_deserializer=node__pb2.grpcModel.FromString,
                    response_serializer=node__pb2.ModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'node.NodeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NodeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetChunk(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/node.NodeService/GetChunk',
            node__pb2.GetChunkRequest.SerializeToString,
            node__pb2.GetChunkResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Verify(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/node.NodeService/Verify',
            node__pb2.VerifyRequest.SerializeToString,
            node__pb2.VerifyResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SaveChunk(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/node.NodeService/SaveChunk',
            node__pb2.SaveChunkRequest.SerializeToString,
            node__pb2.SaveChunkResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeliverBlock(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/node.NodeService/DeliverBlock',
            node__pb2.grpcBlock.SerializeToString,
            node__pb2.BlockResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeliverModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/node.NodeService/DeliverModel',
            node__pb2.grpcModel.SerializeToString,
            node__pb2.ModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
