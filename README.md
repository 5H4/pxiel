# Pxiel
Mini dot have large Node, node have small size, size is a sum of the vector, vector has weight , weight has prediction.

Train and predict with 2 lines of code.

```python
# Train
from no_tf import pxiel as px
from no_tf import nutela
X, y = [], []

# Read csv with pandas or whatever.
# data [category, text]
# for each row do
# Example:
for ....
  nutela.nuty(row.Message.upper())
  
  append X,y


model = px.Pxiel(X = X, y=y)

# Predict 

model = px.Pxiel()
# input is a text
pred = nutela.nuty(input)

prediction = model.Predict(pred)

if prediction == 0:
    print('ham')
else:
    print('spam')

```
