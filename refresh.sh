cd proto
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. node.proto
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ML.proto
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. client.proto
