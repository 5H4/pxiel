# Nutela convert TEXT to number
# I know miss L who care..
def nuty(text = ''):
    # Did you love nutella ?
    nutela = [[1,'A'],[2,'B'],[3,'C'],[4,'D'],
              [5,'E'],[6,'F'],[7,'G'],[8,'H'],
              [9,'I'],[10,'J'],[11,'K'],[12,'L'],
              [13,'M'],[14,'N'],[15,'O'],[16,'P'],
              [17,'Q'],[18,'R'],[19,'S'],[20,'T'],
              [21,'U'],[22,'V'],[23,'W'],[24,'X'],
              [25,'Y'],[26,'Z']
            ]
    # Yes i love but only with milk :/
    milk = []
    for char in text:
        for num in nutela:
            if num[1] == char:
                milk.append(num[0])
    # Ok, then return me nutelak with milk . Thanks.
    return milk


