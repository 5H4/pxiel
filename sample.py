from no_tf import pxiel as px
from no_tf import nutela
import ast
import pandas as pd

#############################
#############################
  # PREPARE DATA & TRAIN #
#############################
#############################
data = pd.read_csv('spam.csv')
X = []
Y = []
for i, row in data.iterrows():
    r = 0
    if row.Category == 'spam':
        r = 1
    li = []
    for char in row.Message.upper():
        for num in nutela:
            if num[1] == char:
                li.append(num[0])
    if len(li) > 3:
        Y.append(r)
        X.append(li)

model = px.Pxiel(X = X, y=Y)

############################
############################
  # LOAD MODEL & PREDICT #
############################
############################
model = px.Pxiel()

input = 'How many pages should a 20 mark essay be, examples of argumentative essay ... model essay how to write a text response essay conclusion mera priya tyohar'.upper()

pred = nutela.nuty(input)

if model.Predict(pred) == 0:
    print('ham')
else:
    print('spam')