import os
import sys

from config import shared_dir, storage_dir

sys.path.insert(0, str(shared_dir))
sys.path.insert(0, str(storage_dir))

from concurrent import futures

import grpc
from dotenv import dotenv_values, load_dotenv
from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import (exin_classifier_gc, exin_translator_gc,
                         protein_translator_gc, sliding_window_tagger_gc)
from generated import data_pb2, data_pb2_grpc

load_dotenv()

def from_request(req):
	annotations_relative_path = req.annotationsPath
	fasta_relative_path = req.fastaPath

	annotations_file_path = os.path.join(storage_dir, annotations_relative_path)
	fasta_file_path = os.path.join(storage_dir, fasta_relative_path)
	request = dict(
		seq_max_len = req.sequenceMaxLength,
		annotations_file_path = annotations_file_path
	)

	if req.fastaPath:
		request.update({
			"fasta_file_path": fasta_file_path
		})
	
	return request

def from_response(res) -> dict[str, str]:
	return dict(
		sequence = res["sequence"],
		target = res["target"],
		flankBefore = res.get("flank_before", ""),
		flankAfter = res.get("flank_after", ""),
		organism = res.get("organism", ""),
		gene = res.get("gene", "")
	)

class ExtractionService(data_pb2_grpc.ExtractionService):
	def ExInClassifierGenbank(self, request, context: grpc.ServicerContext):
		for item in exin_classifier_gb(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

	def ExInTranslatorGenbank(self, request, context: grpc.ServicerContext):
		for item in exin_translator_gb(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))
	
	def SlidingWindowTaggerGenbank(self, request, context: grpc.ServicerContext):
		for item in sliding_window_tagger_gb(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

	def ProteinTranslatorGenbank(self, request, context: grpc.ServicerContext):
		for item in protein_translator_gb(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

	def ExInClassifierGencode(self, request, context: grpc.ServicerContext):
		for item in exin_classifier_gc(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

	def ExInTranslatorGencode(self, request, context: grpc.ServicerContext):
		for item in exin_translator_gc(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))
	
	def SlidingWindowTaggerGencode(self, request, context: grpc.ServicerContext):
		for item in sliding_window_tagger_gc(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

	def ProteinTranslatorGencode(self, request, context: grpc.ServicerContext):
		for item in protein_translator_gc(**from_request(request)):
			yield data_pb2.ExtractionResponse(**from_response(item))

def server():
	config = dotenv_values(".env")
	port = config["PORT"]

	server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
	data_pb2_grpc.add_ExtractionServiceServicer_to_server(ExtractionService(), server)
	server.add_insecure_port(f"[::]:{port}")
	server.start()
	print(f"gRPC Server is running on port {port}...")
	server.wait_for_termination()