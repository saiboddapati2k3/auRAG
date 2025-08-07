import itertools

def batch_vectors(vectors, batch_size=32):
    """
    Creates batches of vectors.
    """
    iterator = iter(vectors)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch
