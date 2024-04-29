from hp5dataset import MultiFileDataset as Dataset
from torch.utils.data import DataLoader 
from tqdm import tqdm

data_dir = "/data2/peter/aiw"
# csv_file = "path/to/your/csvfile.csv"
train_dataset = Dataset(data_dir=data_dir, type="train", csv_file=None, demo=0, decode_type="heatmap")
train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=8)
missing_count = 0
progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))

for i, (text, bound, mask, input2, labels, heats) in progress_bar:
    tmp = i


print(f"Total missing elements: {missing_count}")