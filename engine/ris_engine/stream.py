"""Weights kept in pinned CPU memory and copied to the GPU as they are needed, for GPUs that
cannot hold every model at once (int8.stream_blocks, TextEncoder).

A WeightStream manages "units" of identical layout (DiT blocks, text encoder layers).  The
copies run on their own CUDA stream into a few reusable GPU slots, ahead of the unit being
computed, so they overlap with the computation of the units before them.  The owner announces
each run's unit order with start(); forward hooks on the units then wait for their weights and
hand the slot back afterwards.  Results are bit-identical to keeping the weights on the GPU.
"""

import torch

GiB = 1 << 30
STREAM_SLOTS = 3  # GPU copies of streamed units: the one computing and two arriving


def spread(units, n):
    """n of units, spread evenly (streamed units between resident ones overlap best)."""
    return [units[int((k + 0.5) * len(units) / n)] for k in range(n)] if n > 0 else []


def align(n, to=256):
    return (n + to - 1) // to * to


def tensor_bytes(module):
    return sum(t.nbytes for t in (*module.parameters(), *module.buffers()))


class WeightStream:
    def __init__(self, units, device, slots=STREAM_SLOTS):
        self.copy_stream = torch.cuda.Stream(device)
        tensors = lambda unit: [(owner, name, tensor.data) for owner in unit.modules()
                                for name, tensor in (*owner._parameters.items(), *owner._buffers.items())
                                if tensor is not None]
        # one pinned buffer for all of them: pin_memory() would round each tensor up to a power of two
        size = sum(align(t.nbytes) for unit in units for _, _, t in tensors(unit))
        self.host = torch.empty(size, dtype=torch.uint8)
        torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(self.host.data_ptr(), size, 0))
        self.cpu, offset = {}, 0
        for unit in units:
            refs = []
            for owner, name, tensor in tensors(unit):
                cpu = self.host[offset:offset + tensor.nbytes].view(tensor.dtype).view(tensor.shape)
                cpu.copy_(tensor)
                refs.append((owner, name, cpu))
                offset += align(tensor.nbytes)
            self.cpu[unit] = refs
            self._attach(unit, [torch.empty(0, dtype=cpu.dtype, device=device) for _, _, cpu in refs])
            unit.register_forward_pre_hook(lambda module, args: self.acquire(module))
            unit.register_forward_hook(lambda module, args, out: self.release(module))
        first = self.cpu[units[0]]
        for refs in self.cpu.values():
            if [t.shape for _, _, t in refs] != [t.shape for _, _, t in first]:
                raise ValueError("streamed units must have the same layout")
        self.slots = [[torch.empty_like(cpu, device=device) for _, _, cpu in first] for _ in range(slots)]
        self.idle = [torch.empty(0, dtype=cpu.dtype, device=device) for _, _, cpu in first]
        self.copied = [None] * slots  # event: the slot holds its unit's weights
        self.freed = [None] * slots  # event: the last unit in the slot is done with it
        self.order, self.next_run, self.next_copy = [], 0, 0

    @property
    def unit_bytes(self):
        return sum(t.nbytes for t in self.slots[0])

    @property
    def gpu_bytes(self):
        return len(self.slots) * self.unit_bytes

    def _attach(self, unit, tensors):
        for (owner, name, _), tensor in zip(self.cpu[unit], tensors):
            if name in owner._parameters:
                owner._parameters[name].data = tensor
            else:
                owner._buffers[name] = tensor

    def start(self, order):
        """order: the units this run computes, in order (resident ones are skipped)."""
        self.order = [unit for unit in order if unit in self.cpu]
        self.next_run = self.next_copy = 0
        for _ in self.slots:
            self._copy_next()

    def _copy_next(self):
        if self.next_copy >= len(self.order):
            return
        i = self.next_copy % len(self.slots)
        with torch.cuda.stream(self.copy_stream):
            if self.freed[i] is not None:
                self.copy_stream.wait_event(self.freed[i])
            for slot, (_, _, cpu) in zip(self.slots[i], self.cpu[self.order[self.next_copy]]):
                slot.copy_(cpu, non_blocking=True)
            self.copied[i] = torch.cuda.Event()
            self.copied[i].record(self.copy_stream)
        self.next_copy += 1

    def acquire(self, unit):
        if self.next_run >= len(self.order) or self.order[self.next_run] is not unit:
            raise RuntimeError("streamed weights were used in a different order than announced")
        i = self.next_run % len(self.slots)
        torch.cuda.current_stream().wait_event(self.copied[i])
        self._attach(unit, self.slots[i])

    def release(self, unit):
        i = self.next_run % len(self.slots)
        self.freed[i] = torch.cuda.Event()
        self.freed[i].record(torch.cuda.current_stream())
        self._attach(unit, self.idle)
        self.next_run += 1
        self._copy_next()
