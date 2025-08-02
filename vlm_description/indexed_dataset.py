from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    """Dataset wrapper for slicing large datasets across multiple processes"""
    
    def __init__(self, dataset_source, start_index=0, end_index=9999999):
        super().__init__()
        
        self.dataset_source = dataset_source
        self.start_index = start_index
        self.end_index = min(end_index, len(self.dataset_source))
        
        if self.start_index >= len(self.dataset_source):
            raise ValueError(f"start_index ({self.start_index}) >= dataset length ({len(self.dataset_source)})")
        
        if self.start_index >= self.end_index:
            raise ValueError(f"start_index ({self.start_index}) >= end_index ({self.end_index})")

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        if self.dataset_source is None:
            raise RuntimeError("Dataset source is None")
            
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        
        actual_index = index + self.start_index
        return self.dataset_source[actual_index]