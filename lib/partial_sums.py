from __future__ import division
import numpy as np


class NoisePartialSum:
    def __init__(self, start, size, noise):
        self.start = start
        self.size = size
        self.noise = noise

    def __str__(self):
        return 'NoisePartialSum(start=%i, size=%i)' % (self.start, self.size)


class NoisePartialSumStore:
    def __init__(self, noise_generator, release_method='tree'):
        if release_method != 'tree' and noise_generator.noise_type != 'laplacian':
            raise NotImplementedError
        if release_method not in ['tree', 'every', 'once', 'sqrt', 'hybrid']:
            raise NotImplementedError
        self.noise_generator = noise_generator
        self.release_method = release_method
        self.store = {}
        self.START_TIME = 1

    def is_power_of_two(self, val):
        return ((val & (val - 1)) == 0) and val > 0

    def consolidate_for_once(self, time):
        """Delete all partial sums except at start.

        This is used for the 'once' release method. The first noise added
        will be of a magnitude great enough to protect privacy throughout
        the algorithm.

        Args:
            time (int): time of newly added partial sum
        """
        if time != self.START_TIME:
            del self.store[time]

    def consolidate_for_every(self, time):
        """Collapse all partial sums into one partial sum.

        This is used for the 'every' release method. Each new partial sum
        will have a small amount of noise that will accumulate in a bigger
        and bigger partial sum.

        Args:
            time (int): time of newly added partial sum
        """
        current_total_noise = self.store[self.START_TIME].noise
        newly_added_noise = self.store[time].noise
        new_total_noise = current_total_noise + newly_added_noise
        new_psum_size = time
        self.store[self.START_TIME] = NoisePartialSum(
            1, new_psum_size, new_total_noise)
        if time != self.START_TIME:
            del self.store[time]

    def consolidate_for_sqrt(self, time):
        """Collapse all block partial sums into one partial sum.

        This is used for the 'sqrt' release method. Theoretically, we compute partial sums
        of either a block's size or of a single item. If there are enough single items to
        create a block, consolidate them into one block. To achieve O(1) storage, we
        consolidate all block-level partial sums as well.

        Args:
            time (int): time of newly added partial sum
        """
        eps = self.noise_generator.eps
        block_size = int(np.sqrt(self.noise_generator.T))
        block_start = time - block_size + 1
        if block_start in self.store:
            for _t in range(block_start, time + 1):
                del self.store[_t]
            block_noise = self.noise_generator.laplacian(eps)
            if block_start == self.START_TIME:
                self.store[self.START_TIME] = NoisePartialSum(
                    self.START_TIME, block_size, block_noise)
            else:
                self.store[self.START_TIME] = NoisePartialSum(
                    self.START_TIME, self.store[self.START_TIME].size + block_size, self.store[self.START_TIME].noise + block_noise)

    def consolidate_for_tree(self, time):
        """Collapse all partial sums into "power of two"-sized blocks.

        This is used for the 'tree' release method. Instead of fixed-sized blocks like
        in the 'sqrt' release method, this consolidation technique will collapse sums
        into "power of two"-sized blocks. This method recursively combines equal-sized
        partial sums until there are no more.

        Args:
            time (int): time of newly added partial sum
        """
        prev_p_sum_time = self.store[time].start - self.store[time].size
        if prev_p_sum_time in self.store:
            if self.store[time].size == self.store[prev_p_sum_time].size:
                new_size = self.store[time].size * 2
                eps = self.noise_generator.eps
                delta = self.noise_generator.delta
                T = self.noise_generator.T
                new_noise = self.noise_generator.generate_noise_tree(eps, delta, T)
                self.store[prev_p_sum_time] = NoisePartialSum(
                    prev_p_sum_time, new_size, new_noise)
                del self.store[time]
                self.consolidate_for_tree(prev_p_sum_time)

    def consolidate_for_hybrid(self, time):
        """Collapse all partial sums into one "power of two"-sized block with a tree.

        This is used for the 'hybrid' release method.

        Args:
            time (int): time of newly added partial sum
        """
        if self.is_power_of_two(time) and time > self.START_TIME:
            new_size = time
            new_noise = self.store[self.START_TIME].noise + self.store[time].noise
            self.store = {
                self.START_TIME: NoisePartialSum(self.START_TIME, new_size, new_noise)
            }
        else:
            prev_p_sum_time = self.store[time].start - self.store[time].size
            if prev_p_sum_time in self.store:
                if self.store[time].size == self.store[prev_p_sum_time].size:
                    new_size = self.store[time].size * 2
                    eps = self.noise_generator.eps
                    delta = self.noise_generator.delta
                    T = 2**int(np.log2(time))
                    new_noise = self.noise_generator.generate_noise_tree(eps, delta, T)
                    self.store[prev_p_sum_time] = NoisePartialSum(
                        prev_p_sum_time, new_size, new_noise)
                    del self.store[time]
                    self.consolidate_for_hybrid(prev_p_sum_time)


    def consolidate_store(self, time):
        if self.release_method == 'once':
            self.consolidate_for_once(time)
        elif self.release_method == 'every':
            self.consolidate_for_every(time)
        elif self.release_method == 'sqrt':
            self.consolidate_for_sqrt(time)
        elif self.release_method == 'tree':
            self.consolidate_for_tree(time)
        elif self.release_method == 'hybrid':
            self.consolidate_for_hybrid(time)

    def add_noise(self, time):
        T = self.noise_generator.T
        eps = self.noise_generator.eps
        delta = self.noise_generator.delta
        noise = self.noise_generator.zeros()
        if self.release_method == 'once':
            if len(self.store) == 0:
                noise = self.noise_generator.laplacian(eps / T)
        elif self.release_method == 'every':
            noise = self.noise_generator.laplacian(eps)
        elif self.release_method == 'sqrt':
            noise = self.noise_generator.laplacian(eps / 2)
        elif self.release_method == 'tree':
            noise = self.noise_generator.generate_noise_tree(eps, delta, T)
        elif self.release_method == 'hybrid':
            if self.is_power_of_two(time):
                noise = self.noise_generator.laplacian(2 * eps)
            else:
                time_horizon = 2**int(np.log2(time))
                noise = self.noise_generator.laplacian((eps / 2) / np.log2(time_horizon))
        self.store[time] = NoisePartialSum(start=time, size=1, noise=noise)
        self.consolidate_store(time)

    def release_noise(self):
        """Returns the sum of noise in all partial sums in store."""
        N = self.noise_generator.zeros()
        for p_sum in self.store.values():
            N += p_sum.noise
        return N
