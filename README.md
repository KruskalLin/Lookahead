# Lookahead
Very simple implementation of lookahead optimizer in pytorch.

## Usage

```python
optimizer = Lookahead(optim.Adam(net.parameters(), lr=0.001), k=5, alpha=0.5)
...
optimizer.zero_grad()
...
optimizer.step()
...
```

## Noted

Parameters are shared if there are multiple optimizers so Lookahead class does not implement Optimizer Class. Other optimizer usages can be added.
