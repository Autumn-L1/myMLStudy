---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
---

```python
import torch
import torch.nn as nn
import torchvision
import numpy as np
```

Sigmoid：单类预测到概率
Softmax：多类预测到概率
![[Pasted image 20251020165644.png|300]]
![[Pasted image 20251020164259.png|300]]


![[Pasted image 20251020152304.png]]

```python
def softmax(x):
	return np.exp(x)/np.sum(np.exp(x), axis=0)
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy', outputs)
```

```python
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print('softmax torch', outputs)
```


```python
def cross_entropy(actual, predicted):
	loss = -np.sum(actual * np.log(predicted))
	return loss

# y must be one hot encoded
Y = np.array([1,0,0])

y_pred_good = np.array([0.7,0.2,0.1])
y_pred_bad = np.array([0.1,0.3,0.6])
l1 = cross_entropy(Y,y_pred_good)
l2 = cross_entropy(Y,y_pred_bad)
print(f'Good cross entropy: {l1:.4f}')
print(f'Bad cross entropy: {l2:.4f}')
```

nn.CrossEntropyLoss

![[Pasted image 20251020155201.png]]

```python
loss = nn.CrossEntropyLoss()

# 3 samples, 3 classes
y = torch.tensor([2,0,1])

# n_samples x n_classes = 3*3
y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1],[2.0, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1],[0.1, 3.0, 0.1]])

l1 = loss(y_pred_good,y)
l2 = loss(y_pred_bad,y)
print(f'Good cross entropy: {l1.item():.4f}')
print(f'Bad cross entropy: {l2.item():.4f}')

_,predictions1 = torch.max(y_pred_good,1)
_,predictions2 = torch.max(y_pred_bad,1)
print(predictions1)
print(predictions2)
```

```python
# Multiclass problem
class NeuralNet2(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
	super(NeuralNet2, self).__init__()
	self.linear1 = nn.Linear(input_size, hidden_size)
	self.relu = nn.ReLU()
	self.linear2 = nn.Linear(hidden_size, num_classes)
	
	def forward(self, x):
		out = linear1(x)
		out = relu(out)
		out = linear2(out)
		# 无需softmax转换，因为使用nn.CrossEntropyLoss()计算， 该函数自带softmax转换
		return out 

model = NeuralNet2(input_size=28*28, hidden_size = 5, num_classes = 3)
criterion = nn.CrossEntropyLoss()
```