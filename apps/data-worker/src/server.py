import os
import sys

from config import SHARED_DIR, STORAGE_DIR

sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(STORAGE_DIR))

from concurrent import futures

import grpc
from dotenv import dotenv_values, load_dotenv
from extractor import extract_data
from generated import data_pb2, data_pb2_grpc

load_dotenv()

def from_request(req) -> str:
	annotations_relative_path = req.path

	annotations_file_path = os.path.join(STORAGE_DIR, annotations_relative_path)
	return annotations_file_path

class ExtractionService(data_pb2_grpc.ExtractionService):
	def Extract(
		self,
		request,
		context: grpc.ServicerContext
	):
		try:
			for item in extract_data(from_request(request)):
				cds = [
					data_pb2.CDSRegion(**cds_property) for cds_property in item["cds"]
				]
				exin = [
					data_pb2.ExInRegion(**exin_property) for exin_property in item["exin"]
				]

				response = data_pb2.ExtractionResponse(
					sequence=item["sequence"],
					accession=item["accession"],
					organism=item["organism"],
					cds=cds,
					exin=exin
				)

				yield response
		except Exception as e:
			context.set_code(grpc.StatusCode.INTERNAL)
			context.set_details(f"Internal error: {str(e)}")
			return

def server() -> None:
	config = dotenv_values(".env")
	port = config["PORT"]

	server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
	data_pb2_grpc.add_ExtractionServiceServicer_to_server(ExtractionService(), server)
	server.add_insecure_port(f"[::]:{port}")
	server.start()
	print(f"gRPC Server is running on port {port}...")
	server.wait_for_termination()