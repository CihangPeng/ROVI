import os
import json
from torch.utils.data import Dataset
from pathlib import Path

def collect_json_files(folders):
    """Collect all JSON files from specified folders."""
    json_files = []
    
    for folder in folders:
        if os.path.isdir(folder):
            print(f'Loading dataset from {folder}')
            json_files.extend([
                os.path.join(folder, filename)
                for filename in os.listdir(folder)
                if filename.endswith('.json')
            ])
        else:
            print(f'Dataset directory does not exist: {folder}')
            exit()
            
    return json_files

class JsonDataset(Dataset):
    """Dataset for loading JSON files."""
    
    def __init__(self, folders):
        self.folders = folders
        self.json_files = collect_json_files(folders)
            
    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        
        path_bundle = Path(json_file)
        directory = path_bundle.parent
        filename = path_bundle.name
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract filename without extension as fallback key
        filename_without_ext = Path(filename).stem
        
        if 'new_key' not in data.keys():
            data['new_key'] = filename_without_ext
            
        return data

def collate_fn(batch):
    """Collate function for batching JSON data."""
    new_key_list = []
    web_caption_list = []
    vlm_caption_list = []
    sample_list = []
    
    for sample in batch:
        new_key_list.append(sample['new_key'])
        web_caption_list.append(sample['web_caption'])
        vlm_caption_list.append(sample['vlm_caption'])
        sample_list.append(sample)
    
    result = {
        'sample_list': sample_list,
        'new_key_list': new_key_list,
        'web_caption_list': web_caption_list,
        'vlm_caption_list': vlm_caption_list
    }
    
    # Include phase1 attributes if available (for phase 2 processing)
    if 'llama3_phase1_attributes' in batch[0].keys():
        llama3_phase1_attributes_list = [
            sample['llama3_phase1_attributes'] for sample in batch
        ]
        result['llama3_phase1_attributes_list'] = llama3_phase1_attributes_list
        
    return result

class IndexedDataset(Dataset):
    """Dataset with distributed processing support via data slicing."""
    
    def __init__(self, folders, gpu_num=1, data_slice_index=0, test_cutdown=9999999):
        super().__init__()

        self.dataset_source = JsonDataset(folders)
        
        # Calculate data slicing for distributed processing
        total_num = len(self.dataset_source)
        divide = total_num // gpu_num
        start_end = {}

        for li in range(gpu_num):
            if li != gpu_num - 1:
                start_end[str(li)] = [li * divide, (li + 1) * divide]
            else:
                start_end[str(li)] = [li * divide, total_num]

        start_index, end_index = start_end[str(data_slice_index)]
        end_index = min(end_index, start_index + test_cutdown)
        
        print(f'Subset {data_slice_index} created: len={end_index - start_index}/{total_num}, '
              f'start={start_index}, end={end_index}')
        
        self.start_index = start_index
        self.end_index = min(end_index, len(self.dataset_source))

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        assert self.dataset_source is not None

        if index >= self.end_index - self.start_index:
            raise IndexError 
        
        return self.dataset_source[index + self.start_index]