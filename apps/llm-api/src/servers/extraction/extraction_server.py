from concurrent import futures

import grpc
from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import (exin_classifier_gc, exin_translator_gc,
                         protein_translator_gc, sliding_window_tagger_gc)
from servers.generated import extraction_pb2, extraction_pb2_grpc


def from_request(req):
	request = dict(
		seq_max_len = req.sequenceMaxLength,
		annotations_file_path = req.annotationsPath,
	)

	if req.fastaPath:
		request.update({
			"fasta_file_path": req.fastaPath
		})
	
	return request

def from_response(res):
	return dict(
		sequence = res.sequence,
		target = res.target,
		flankBefore = res.flank_before,
		flankAfter = res.flank_after,
		organism = res.organism,
		gene = res.gene
	)

class ExtractionService(extraction_pb2_grpc.ExtractionService):
	def ExInClassifierGenbank(self, request, context: grpc.ServicerContext):
		for item in exin_classifier_gb(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

	def ExInTranslatorGenbank(self, request, context: grpc.ServicerContext):
		for item in exin_translator_gb(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))
	
	def SlidingWindowTaggerGenbank(self, request, context: grpc.ServicerContext):
		for item in sliding_window_tagger_gb(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

	def ProteinTranslatorGenbank(self, request, context: grpc.ServicerContext):
		for item in protein_translator_gb(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

	def ExInClassifierGencode(self, request, context: grpc.ServicerContext):
		for item in exin_classifier_gc(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

	def ExInTranslatorGencode(self, request, context: grpc.ServicerContext):
		for item in exin_translator_gc(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))
	
	def SlidingWindowTaggerGencode(self, request, context: grpc.ServicerContext):
		for item in sliding_window_tagger_gc(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

	def ProteinTranslatorGencode(self, request, context: grpc.ServicerContext):
		for item in protein_translator_gc(**from_request(request)):
			yield extraction_pb2.ExtractionResponse(**from_response(item))

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	extraction_pb2_grpc.add_ExtractionServiceServicer_to_server(ExtractionService(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	print("gRPC Server is running on port 50051...")
	server.wait_for_termination()