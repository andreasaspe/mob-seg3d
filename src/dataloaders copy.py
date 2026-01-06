from torch.utils.data import Dataset, DataLoader
import os 
import pickle 
import torch
from tqdm import tqdm

class HackyEvalLoader(Dataset):

    def __init__(self, path_to_data, old_eval_set = None, recreate = False):
        
        if recreate:
            self.delete_potential_dataset(path_to_data)
            self.create_dataset(path_to_data, old_eval_set)

            assert old_eval_set is not None
        
        self.path_to_data = path_to_data
        self.data_files = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data) if file.endswith('pkl')]


    def delete_potential_dataset(self, path_to_data):

        if not os.path.exists(path_to_data):
            return None
        
        currently_existing_files = [os.path.join(path_to_data, file) for file in os.listdir(path_to_data) if file.endswith('.pkl')]
        for file in currently_existing_files:
            os.remove(file)
        
        print("Removed ", len(currently_existing_files), 'Files from the dataset directory')
    
    def create_dataset(self, path_to_data, old_eval_set):
        
        if not os.path.isdir(path_to_data):
            os.makedirs(path_to_data)
        
        identifiers = old_eval_set.generator._data.identifiers
        has_been_saved = {key: False for key in identifiers}

        pbar = tqdm(total = len(identifiers), desc = 'Saving dataset')

        while not all(has_been_saved.values()):
            
            elem = next(iter(old_eval_set))
            key = elem['keys'][0].item()
            if has_been_saved[key]:
                continue
            filename = os.path.join(path_to_data, key+".pkl")

            with open(filename, 'wb') as handle:
                pickle.dump(elem, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            has_been_saved[key] = True
            pbar.update(1)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        elem = pickle.load(open(self.data_files[index], 'rb'))
        return elem



