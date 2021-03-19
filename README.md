# Pxiel
Mini dot have large Node, node have small size, size is a sum of the vector, vector has weight , weight has prediction.


```python
# Train
from no_tf import pxiel as px
X, y = [], []

model = px.Pxiel(X = X, y=y, elastic = True)

# Predict 

model = px.Pxiel()

prediction = model.Predict([1, 3, 4, 2])
```
