from typing import Generator, TypeVar


def chunked_generator(generator, size):
  batch = []
  for item in generator:
    batch.append(item)
    if len(batch) == size:
      yield batch
      batch = []
  if batch:
    yield batch

T = TypeVar('T')

def batch_list(data: list[T], batch_size: int) -> Generator[list[T], None, None]:
  for i in range(0, len(data), batch_size):
    yield data[i:i + batch_size]