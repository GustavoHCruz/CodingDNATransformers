def chunked_generator(generator, size):
  batch = []
  for item in generator:
    batch.append(item)
    if len(batch) == size:
      yield batch
      batch = []
  if batch:
    yield batch