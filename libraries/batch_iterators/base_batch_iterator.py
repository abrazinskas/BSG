from multiprocessing import Process, Queue


class BaseBatchIterator():

    def __init__(self):
        pass

    def __iter__(self):
        """
        Separate process pre-loading of data and iteration over its batches.

        """
        process, queue = self.__i_parallel_load_data_batches()
        process.start()
        process.deamon = True
        while True:
            batch = queue.get()
            if batch is None:
                process.join()
                break  # the file has ended
            yield batch

    def load_data_batches_to_queue(self, queue):
        raise NotImplementedError  # this has to be assigned in a subclass

    def __i_parallel_load_data_batches(self, queue_size=5):
        """
        Loads batches on a separate process.

        """
        queue = Queue(queue_size)
        process = Process(target=self.load_data_batches_to_queue, args=(queue, ))
        return process, queue

