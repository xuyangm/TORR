# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: client.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63lient.proto\x12\x06\x63lient\"+\n\x0bGlobalModel\x12\r\n\x05round\x18\x01 \x01(\x05\x12\r\n\x05model\x18\x02 \x01(\x0c\"\x1b\n\nLocalModel\x12\r\n\x05model\x18\x01 \x01(\x0c\x32H\n\rClientService\x12\x37\n\nLocalTrain\x12\x13.client.GlobalModel\x1a\x12.client.LocalModel\"\x00\x62\x06proto3')



_GLOBALMODEL = DESCRIPTOR.message_types_by_name['GlobalModel']
_LOCALMODEL = DESCRIPTOR.message_types_by_name['LocalModel']
GlobalModel = _reflection.GeneratedProtocolMessageType('GlobalModel', (_message.Message,), {
  'DESCRIPTOR' : _GLOBALMODEL,
  '__module__' : 'client_pb2'
  # @@protoc_insertion_point(class_scope:client.GlobalModel)
  })
_sym_db.RegisterMessage(GlobalModel)

LocalModel = _reflection.GeneratedProtocolMessageType('LocalModel', (_message.Message,), {
  'DESCRIPTOR' : _LOCALMODEL,
  '__module__' : 'client_pb2'
  # @@protoc_insertion_point(class_scope:client.LocalModel)
  })
_sym_db.RegisterMessage(LocalModel)

_CLIENTSERVICE = DESCRIPTOR.services_by_name['ClientService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GLOBALMODEL._serialized_start=24
  _GLOBALMODEL._serialized_end=67
  _LOCALMODEL._serialized_start=69
  _LOCALMODEL._serialized_end=96
  _CLIENTSERVICE._serialized_start=98
  _CLIENTSERVICE._serialized_end=170
# @@protoc_insertion_point(module_scope)