from no_tf import pxiel as px

X = []
y = []

import random
for x in range(0, 5000):
    randomlist = random.sample(range(0, 5), 4)
    X.append(randomlist)
    y.append(random.sample(range(0, 2), 1)[0])

model = px.Pxiel(X = X, y=y, elastic = True)

model = px.Pxiel()

prediction = model.Predict([1, 3, 4, 2])

print(prediction)