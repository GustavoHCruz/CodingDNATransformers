from concurrent import futures

import grpc
from etl.genbank import (exin_classifier_gb, exin_translator_gb,
                         protein_translator_gb, sliding_window_tagger_gb)
from etl.gencode import (exin_classifier_gc, exin_translator_gc,
                         protein_translator_gc, sliding_window_tagger_gc)
from generated import extraction_pb2, extraction_pb2_grpc


class ExtractionService(extraction_pb2_grpc.ExtractionService):
	def ExInClassifierGenbank(self, request, context):
		for item in exin_classifier_gb(**request):
			yield extraction_pb2.ExtractionResponse(**item)

	def ExInTranslatorGenbank(self, request, context):
		for item in exin_translator_gb(**request):
			yield extraction_pb2.ExtractionResponse(**item)
	
	def SlidingWindowTaggerGenbank(self, request, context):
		for item in sliding_window_tagger_gb(**request):
			yield extraction_pb2.ExtractionResponse(**item)

	def ProteinTranslatorGenbank(self, request, context):
		for item in protein_translator_gb(**request):
			yield extraction_pb2.ExtractionResponse(**item)

	def ExInClassifierGencode(self, request, context):
		for item in exin_classifier_gc(**request):
			yield extraction_pb2.ExtractionResponse(**item)

	def ExInTranslatorGencode(self, request, context):
		for item in exin_translator_gc(**request):
			yield extraction_pb2.ExtractionResponse(**item)
	
	def SlidingWindowTaggerGencode(self, request, context):
		for item in sliding_window_tagger_gc(**request):
			yield extraction_pb2.ExtractionResponse(**item)

	def ProteinTranslatorGencode(self, request, context):
		for item in protein_translator_gc(**request):
			yield extraction_pb2.ExtractionResponse(**item)

def serve():
		server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
		extraction_pb2_grpc.add_ExtractionServiceServicer_to_server(ExtractionService(), server)
		server.add_insecure_port('[::]:50051')
		server.start()
		print("gRPC Server is running on port 50051...")
		server.wait_for_termination()

if __name__ == '__main__':
		serve()
