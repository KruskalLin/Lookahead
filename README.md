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
