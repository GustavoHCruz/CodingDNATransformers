import sys

from config import SHARED_DIR, STORAGE_DIR

sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(STORAGE_DIR))

from concurrent import futures

import grpc
from dotenv import dotenv_values, load_dotenv
from generated import llm_pb2_grpc
from google.protobuf import empty_pb2
from llms.protein_translator.gpt import create_model as pt_gpt_create_model
from llms.protein_translator.gpt import train_model as pt_gpt_train_model
from redis_service import RedisService

load_dotenv()
redis = RedisService()

class LlmService(llm_pb2_grpc.LlmService):
	def CreateModel(self, request, context: grpc.ServicerContext) -> empty_pb2.Empty:
		if request.approach == "PROTEINTRANSLATOR":
			if request.checkpoint in ["gpt2", "gpt2-medium"]:
				pt_gpt_create_model(
					checkpoint=request.checkpoint,
					name=request.name,
					uuid=request.uuid,
					is_child=request.isChild
				)

		return empty_pb2.Empty()

	def TrainModel(self, request, context: grpc.ServicerContext) -> empty_pb2.Empty:
		if request.approach == "PROTEINTRANSLATOR":
			if request.checkpoint in ["gpt2", "gpt2-medium"]:
				print(request)
				try:
					pt_gpt_train_model(
						name=request.name,
						uuid=request.uuid,
						data_length=request.dataLength,
						epochs=request.epochs,
						batch_size=request.batchSize,
						gradient_accumulation=request.gradientAccumulation,
						lr=request.lr,
						warmup_ratio=request.warmupRatio,
						seed=request.seed
					)
				except Exception as e:
					print(e)

		return empty_pb2.Empty()
	
	def EvalModel(self, request, context: grpc.ServicerContext) -> empty_pb2.Empty:
		print(request)
		return empty_pb2.Empty()

	def Predict(self, request, context: grpc.ServicerContext) -> empty_pb2.Empty:
		print(request)
		return empty_pb2.Empty()

def server():
	config = dotenv_values(".env")
	port = config["PORT"]

	server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
	llm_pb2_grpc.add_LlmServiceServicer_to_server(LlmService(), server)
	server.add_insecure_port(f"[::]:{port}")
	server.start()
	print(f"gRPC Server is running on port {port}...")
	server.wait_for_termination()