---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
---
```python
import torch
```

```python
x = torch.tensor([1.,2.,1.], requires_grad = True)
```

```python
y = x + 2
print(y)
```
Calculation Graph (for back propagation)
![[Pasted image 20251016152421.png]]

```python
z = y*y*2
print(z)
```

```python
v = torch.tensor([1,1,1])
z.backward(v) #dz/dx
print(x.grad)

```


```python
x = torch.tensor([1.,2.,1.], requires_grad = True)
print(x)

#下面两行是等价的
x.requires_grad_(False)
# x.detach()

print(x)
y = x + 2
print(y)
```

```python
x = torch.tensor([1.,2.,1.], requires_grad = True)
print(x)

with torch.no_grad():
	print(x)
	y = x + 2
	print(y)
```

调用.backward()计算出的梯度会累加在.grad参数中

```python
weights = torch.ones(4, requires_grad = True)
for epoch in range(3):
	model_output = (weights*3).sum()
	model_output.backward()
	print(f"epoch {epoch}: ",weights.grad)
```

使用.grad.zero_()方法重置梯度
```python
weights = torch.ones(4, requires_grad = True)
for epoch in range(3):
	model_output = (weights*3).sum()
	model_output.backward()
	print(f"epoch {epoch}: ",weights.grad)
	weights.grad.zero_()
```



---此部分在之后的课程中才会讲到
```python
weights = torch.ones(4, requires_grad = True)
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
print(weight.grad)
optimizer.zerograd()
print(weight.grad)
```
