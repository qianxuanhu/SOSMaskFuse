import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_inf, self.next_gt,_,_ = next(self.loader)
        except StopIteration:
            self.next_inf = None
            self.next_gt = None
            return

        with torch.cuda.stream(self.stream):
            self.next_inf = self.next_inf.cuda(non_blocking=True).float()
            self.next_gt = self.next_gt.cuda(non_blocking=True).float()
            #self.next_inf = self.next_inf   # if need
            #self.next_gt = self.next_gt     # if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inf = self.next_inf
        gt = self.next_gt
        self.preload()
        return inf, gt
