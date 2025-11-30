---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.17.3
---

```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

import os
os.chdir('C:/Users/Errorwaf/Nutstore/1/我的坚果云/Obsidian/Knowledge Vault/Knowledge Vault/LLMStudy/PyTorch')
```


```python
class WineDataset(Dataset):
	def __init__(self):
		#data loading
		xy = np.loadtxt('./_asset/data/wine.csv', delimiter = ",", dtype=np.float32, skiprows=1)
		self.x=torch.from_numpy(xy[:,1:])
		self.y=torch.from_numpy(xy[:,[0]]) #array of n_sample, 1
		self.n_samples = xy.shape[0]
		
	def __getitem__(self,index):
		return self.x[index], self.y[index]
	
	def __len__(self):
		return self.n_samples
	
dataset = WineDataset()
first_data = dataset[0]
features, label = first_data
print(features, label)

```

```python
#dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)#在JupyterNotebook运行多线程会报错
dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

dataiter =iter(dataloader)
data = next(dataiter)
features, label = data
print(features, label)
```

```python
# Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples,n_iterations)

for epoch in range(num_epochs):
	for i, (inputs,labels) in enumerate(dataloader):
		if (i+1)%5==0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
```

```python
torchcision.dataset.MNIST()
```