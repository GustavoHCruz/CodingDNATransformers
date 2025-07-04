import sys

from config import SHARED_DIR, STORAGE_DIR

sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(STORAGE_DIR))

from concurrent import futures

import grpc
from dotenv import dotenv_values, load_dotenv
from generated import llm_pb2_grpc
from google.protobuf import empty_pb2
from redis_service import RedisService

load_dotenv()
redis = RedisService()

class LlmService(llm_pb2_grpc.LlmService):
	def CreateModel(self, request, context: grpc.ServicerContext):
		return empty_pb2.Empty()

	def TrainModel(self, request, context: grpc.ServicerContext):
		return empty_pb2.Empty()
	
	def EvalModel(self, request, context: grpc.ServicerContext):
		return empty_pb2.Empty()

	def Predict(self, request, context: grpc.ServicerContext):
		return empty_pb2.Empty()

def server():
	config = dotenv_values(".env")
	port = config["PORT"]

	server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
	llm_pb2_grpc.add_LlmServiceServicer_to_server(LlmService(), server)
	server.add_insecure_port(f"[::]:{port}")
	server.start()
	print(f"gRPC Server is running on port {port}...")
	server.wait_for_termination()