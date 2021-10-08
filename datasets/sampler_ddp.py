from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
import torch.distributed as dist
_LOCAL_PROCESS_GROUP = None
import torch
import pickle

def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class RandomIdentitySampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, mini_batch_size, num_instances):
        self.data_source = data_source
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = mini_batch_size
        self.num_pids_per_batch = (mini_batch_size * self.world_size) // num_instances
        self.num_pids_per_gpu = mini_batch_size // num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        final_idxs = self.sample_list()
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        print(final_idxs[:10])
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        return final_idxs

    def sample_list(self):
        avai_pids = copy.deepcopy(self.pids)
        # for single gpu
        num_iter_per_epoch = float(len(avai_pids)/self.num_pids_per_gpu)
        if num_iter_per_epoch < 1:
            num_iter_per_epoch = 1
        num_pids_per_epoch = int(num_iter_per_epoch*self.num_pids_per_gpu)

        batch_index_list = []
        pid_count = 0
        while True:
            indices = torch.randperm(len(avai_pids))
            for i in indices:
                if pid_count > num_pids_per_epoch:
                    continue
                pid = avai_pids[i]
                t = self.index_dic[pid]
                replace = False if len(t) >= self.num_instances else True
                t = np.random.choice(t, size=self.num_instances, replace=replace).tolist()
                pid_count += 1
                batch_index_list += t
            if pid_count > num_pids_per_epoch:
                break
        return batch_index_list

    def __len__(self):
        return self.length

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

class ImageUniformSampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, mini_batch_size, num_instances):
        self.data_source = data_source
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = mini_batch_size
        # self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.num_pids_per_batch = (mini_batch_size * self.world_size) // num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        #self.world_size = dist.get_world_size()
        self.length //= self.world_size

    def __iter__(self):
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        length = int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        #final_idxs = final_idxs[self.rank * length:(self.rank + 1) * length]
        final_idxs = self.__fetch_current_node_idxs(final_idxs, length)
        self.length = len(final_idxs)
        return iter(final_idxs)


    def __fetch_current_node_idxs(self, final_idxs, length):
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        """
        The long tail data will overwhelm last batches.
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(self.mini_batch_size * self.rank + self.mini_batch_size * i, min(self.mini_batch_size * self.rank + self.mini_batch_size * (i+1), total_num))
            index_target.extend(index)
        index_target_npy = np.array(index_target)
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        """
        cache_index = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = np.arange(self.mini_batch_size * self.rank + self.mini_batch_size * i, min(self.mini_batch_size * self.rank + self.mini_batch_size * (i+1), total_num))
            cache_index.append(index)
        cache_index = np.array(cache_index)
        np.random.shuffle(cache_index)
        index_target_npy = cache_index.flatten()
        final_idxs = list(np.array(final_idxs)[index_target_npy])
        return final_idxs


    def sample_list(self):
        #np.random.seed(self._seed)
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        return self.length


class GLDSampler_DDP(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, mini_batch_size, num_instances):
        self.data_source = data_source
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = mini_batch_size
        self.num_pids_per_batch = (mini_batch_size * self.world_size) // num_instances
        self.num_pids_per_gpu = mini_batch_size // num_instances
        self.clean_continent_dic = defaultdict(list)
        self.noisy_continent_dic = defaultdict(list)

        for index, (_, pid, noisy_flag, continent) in enumerate(self.data_source):
            if not noisy_flag:
                self.clean_continent_dic[continent].append(index)
            else:
                self.noisy_continent_dic[continent].append(index)
        self.length = len(self.data_source) / self.world_size
        self.length = int(self.length)
        self.continent_ratio = {'Asia': 0.5, 'Europe': 0.2, 'Africa': 0.15,
                                'North America': 0.1, 'South America': 0.02,
                                'Antarctica': 0.01, 'Oceania': 0.01, 'OTHER': 0.01}
        self.continent_list = []
        self.continent_prob = []
        for name, prob in self.continent_ratio.items():
            self.continent_list.append(name)
            self.continent_prob.append(prob)

    def __iter__(self):
        final_idxs = self.sample_list()
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        print(final_idxs[:10])
        total_num = len(final_idxs)
        block_num = (length // self.mini_batch_size)
        index_target = []
        return final_idxs

    def sample_list(self):

        batch_index_list = []
        for continent in self.continent_list:
            continent_num = int(self.length * self.continent_ratio[continent])
            clean_num = int(continent_num * 0.66)
            noisy_num = continent_num - clean_num
            clean_data = self.clean_continent_dic[continent]
            noisy_data = self.noisy_continent_dic[continent]
            clean_t = np.random.choice(clean_data, size=clean_num, replace=True).tolist()
            noisy_t = np.random.choice(noisy_data, size=noisy_num, replace=True).tolist()
            batch_index_list += clean_t
            batch_index_list += noisy_t
        np.random.shuffle(batch_index_list)
        return batch_index_list

    def __len__(self):
        return self.length
