# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: node.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nnode.proto\x12\x04node\"\x1f\n\rBlockResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\"\x1d\n\rVerifyRequest\x12\x0c\n\x04hash\x18\x01 \x01(\t\"\x81\x01\n\x0eVerifyResponse\x12\x0e\n\x06result\x18\x01 \x01(\t\x12\x30\n\x06scores\x18\x02 \x03(\x0b\x32 .node.VerifyResponse.ScoresEntry\x1a-\n\x0bScoresEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"X\n\x10SaveChunkRequest\x12\x12\n\nmodel_hash\x18\x01 \x01(\t\x12\x12\n\nchunk_hash\x18\x02 \x01(\t\x12\x0b\n\x03\x62tl\x18\x03 \x01(\x05\x12\x0f\n\x07\x63ontent\x18\x04 \x01(\x0c\"7\n\x11SaveChunkResponse\x12\x12\n\nmodel_hash\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x01(\t\"8\n\tgrpcChunk\x12\x0c\n\x04hash\x18\x01 \x01(\t\x12\r\n\x05index\x18\x02 \x01(\x05\x12\x0e\n\x06keeper\x18\x03 \x01(\t\"\xf7\x01\n\tgrpcBlock\x12\x11\n\ttimestamp\x18\x01 \x01(\x02\x12\x11\n\ttime_diff\x18\x02 \x01(\x02\x12\n\n\x02rd\x18\x03 \x01(\x05\x12\n\n\x02id\x18\x04 \x01(\x05\x12\r\n\x05miner\x18\x05 \x01(\t\x12\x13\n\x0b\x62\x65ta_string\x18\x06 \x01(\x0c\x12)\n\x05stake\x18\x07 \x03(\x0b\x32\x1a.node.grpcBlock.StakeEntry\x12\x1f\n\x06models\x18\x08 \x03(\x0b\x32\x0f.node.grpcModel\x12\x0e\n\x06scores\x18\t \x03(\x02\x1a,\n\nStakeEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\xb7\x01\n\tgrpcModel\x12\r\n\x05owner\x18\x01 \x01(\t\x12\n\n\x02rd\x18\x02 \x01(\x05\x12\x12\n\nmodel_hash\x18\x03 \x01(\t\x12+\n\x06scores\x18\x04 \x03(\x0b\x32\x1b.node.grpcModel.ScoresEntry\x12\x1f\n\x06\x63hunks\x18\x05 \x03(\x0b\x32\x0f.node.grpcChunk\x1a-\n\x0bScoresEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\x1f\n\x0fGetChunkRequest\x12\x0c\n\x04hash\x18\x01 \x01(\t\"#\n\x10GetChunkResponse\x12\x0f\n\x07\x63ontent\x18\x01 \x01(\x0c\"\x1f\n\rModelResponse\x12\x0e\n\x06result\x18\x01 \x01(\t2\xb1\x02\n\x0bNodeService\x12;\n\x08GetChunk\x12\x15.node.GetChunkRequest\x1a\x16.node.GetChunkResponse\"\x00\x12\x35\n\x06Verify\x12\x13.node.VerifyRequest\x1a\x14.node.VerifyResponse\"\x00\x12>\n\tSaveChunk\x12\x16.node.SaveChunkRequest\x1a\x17.node.SaveChunkResponse\"\x00\x12\x36\n\x0c\x44\x65liverBlock\x12\x0f.node.grpcBlock\x1a\x13.node.BlockResponse\"\x00\x12\x36\n\x0c\x44\x65liverModel\x12\x0f.node.grpcModel\x1a\x13.node.ModelResponse\"\x00\x62\x06proto3')



_BLOCKRESPONSE = DESCRIPTOR.message_types_by_name['BlockResponse']
_VERIFYREQUEST = DESCRIPTOR.message_types_by_name['VerifyRequest']
_VERIFYRESPONSE = DESCRIPTOR.message_types_by_name['VerifyResponse']
_VERIFYRESPONSE_SCORESENTRY = _VERIFYRESPONSE.nested_types_by_name['ScoresEntry']
_SAVECHUNKREQUEST = DESCRIPTOR.message_types_by_name['SaveChunkRequest']
_SAVECHUNKRESPONSE = DESCRIPTOR.message_types_by_name['SaveChunkResponse']
_GRPCCHUNK = DESCRIPTOR.message_types_by_name['grpcChunk']
_GRPCBLOCK = DESCRIPTOR.message_types_by_name['grpcBlock']
_GRPCBLOCK_STAKEENTRY = _GRPCBLOCK.nested_types_by_name['StakeEntry']
_GRPCMODEL = DESCRIPTOR.message_types_by_name['grpcModel']
_GRPCMODEL_SCORESENTRY = _GRPCMODEL.nested_types_by_name['ScoresEntry']
_GETCHUNKREQUEST = DESCRIPTOR.message_types_by_name['GetChunkRequest']
_GETCHUNKRESPONSE = DESCRIPTOR.message_types_by_name['GetChunkResponse']
_MODELRESPONSE = DESCRIPTOR.message_types_by_name['ModelResponse']
BlockResponse = _reflection.GeneratedProtocolMessageType('BlockResponse', (_message.Message,), {
  'DESCRIPTOR' : _BLOCKRESPONSE,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.BlockResponse)
  })
_sym_db.RegisterMessage(BlockResponse)

VerifyRequest = _reflection.GeneratedProtocolMessageType('VerifyRequest', (_message.Message,), {
  'DESCRIPTOR' : _VERIFYREQUEST,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.VerifyRequest)
  })
_sym_db.RegisterMessage(VerifyRequest)

VerifyResponse = _reflection.GeneratedProtocolMessageType('VerifyResponse', (_message.Message,), {

  'ScoresEntry' : _reflection.GeneratedProtocolMessageType('ScoresEntry', (_message.Message,), {
    'DESCRIPTOR' : _VERIFYRESPONSE_SCORESENTRY,
    '__module__' : 'node_pb2'
    # @@protoc_insertion_point(class_scope:node.VerifyResponse.ScoresEntry)
    })
  ,
  'DESCRIPTOR' : _VERIFYRESPONSE,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.VerifyResponse)
  })
_sym_db.RegisterMessage(VerifyResponse)
_sym_db.RegisterMessage(VerifyResponse.ScoresEntry)

SaveChunkRequest = _reflection.GeneratedProtocolMessageType('SaveChunkRequest', (_message.Message,), {
  'DESCRIPTOR' : _SAVECHUNKREQUEST,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.SaveChunkRequest)
  })
_sym_db.RegisterMessage(SaveChunkRequest)

SaveChunkResponse = _reflection.GeneratedProtocolMessageType('SaveChunkResponse', (_message.Message,), {
  'DESCRIPTOR' : _SAVECHUNKRESPONSE,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.SaveChunkResponse)
  })
_sym_db.RegisterMessage(SaveChunkResponse)

grpcChunk = _reflection.GeneratedProtocolMessageType('grpcChunk', (_message.Message,), {
  'DESCRIPTOR' : _GRPCCHUNK,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.grpcChunk)
  })
_sym_db.RegisterMessage(grpcChunk)

grpcBlock = _reflection.GeneratedProtocolMessageType('grpcBlock', (_message.Message,), {

  'StakeEntry' : _reflection.GeneratedProtocolMessageType('StakeEntry', (_message.Message,), {
    'DESCRIPTOR' : _GRPCBLOCK_STAKEENTRY,
    '__module__' : 'node_pb2'
    # @@protoc_insertion_point(class_scope:node.grpcBlock.StakeEntry)
    })
  ,
  'DESCRIPTOR' : _GRPCBLOCK,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.grpcBlock)
  })
_sym_db.RegisterMessage(grpcBlock)
_sym_db.RegisterMessage(grpcBlock.StakeEntry)

grpcModel = _reflection.GeneratedProtocolMessageType('grpcModel', (_message.Message,), {

  'ScoresEntry' : _reflection.GeneratedProtocolMessageType('ScoresEntry', (_message.Message,), {
    'DESCRIPTOR' : _GRPCMODEL_SCORESENTRY,
    '__module__' : 'node_pb2'
    # @@protoc_insertion_point(class_scope:node.grpcModel.ScoresEntry)
    })
  ,
  'DESCRIPTOR' : _GRPCMODEL,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.grpcModel)
  })
_sym_db.RegisterMessage(grpcModel)
_sym_db.RegisterMessage(grpcModel.ScoresEntry)

GetChunkRequest = _reflection.GeneratedProtocolMessageType('GetChunkRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETCHUNKREQUEST,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.GetChunkRequest)
  })
_sym_db.RegisterMessage(GetChunkRequest)

GetChunkResponse = _reflection.GeneratedProtocolMessageType('GetChunkResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETCHUNKRESPONSE,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.GetChunkResponse)
  })
_sym_db.RegisterMessage(GetChunkResponse)

ModelResponse = _reflection.GeneratedProtocolMessageType('ModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _MODELRESPONSE,
  '__module__' : 'node_pb2'
  # @@protoc_insertion_point(class_scope:node.ModelResponse)
  })
_sym_db.RegisterMessage(ModelResponse)

_NODESERVICE = DESCRIPTOR.services_by_name['NodeService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _VERIFYRESPONSE_SCORESENTRY._options = None
  _VERIFYRESPONSE_SCORESENTRY._serialized_options = b'8\001'
  _GRPCBLOCK_STAKEENTRY._options = None
  _GRPCBLOCK_STAKEENTRY._serialized_options = b'8\001'
  _GRPCMODEL_SCORESENTRY._options = None
  _GRPCMODEL_SCORESENTRY._serialized_options = b'8\001'
  _BLOCKRESPONSE._serialized_start=20
  _BLOCKRESPONSE._serialized_end=51
  _VERIFYREQUEST._serialized_start=53
  _VERIFYREQUEST._serialized_end=82
  _VERIFYRESPONSE._serialized_start=85
  _VERIFYRESPONSE._serialized_end=214
  _VERIFYRESPONSE_SCORESENTRY._serialized_start=169
  _VERIFYRESPONSE_SCORESENTRY._serialized_end=214
  _SAVECHUNKREQUEST._serialized_start=216
  _SAVECHUNKREQUEST._serialized_end=304
  _SAVECHUNKRESPONSE._serialized_start=306
  _SAVECHUNKRESPONSE._serialized_end=361
  _GRPCCHUNK._serialized_start=363
  _GRPCCHUNK._serialized_end=419
  _GRPCBLOCK._serialized_start=422
  _GRPCBLOCK._serialized_end=669
  _GRPCBLOCK_STAKEENTRY._serialized_start=625
  _GRPCBLOCK_STAKEENTRY._serialized_end=669
  _GRPCMODEL._serialized_start=672
  _GRPCMODEL._serialized_end=855
  _GRPCMODEL_SCORESENTRY._serialized_start=169
  _GRPCMODEL_SCORESENTRY._serialized_end=214
  _GETCHUNKREQUEST._serialized_start=857
  _GETCHUNKREQUEST._serialized_end=888
  _GETCHUNKRESPONSE._serialized_start=890
  _GETCHUNKRESPONSE._serialized_end=925
  _MODELRESPONSE._serialized_start=927
  _MODELRESPONSE._serialized_end=958
  _NODESERVICE._serialized_start=961
  _NODESERVICE._serialized_end=1266
# @@protoc_insertion_point(module_scope)
