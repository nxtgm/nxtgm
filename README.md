# nxtgm
nxtgm next graphical models -- a fun project just for the fun of coding








# design

## Constraints

* we could just return infiniy in the arrays / functions for infesible labels
  but that will be hard to optimize with move making algorithms (or gradient descent in case of continuous variables)

* each value - table  has a boolean "is_constraint" to indicate the present of any constraints

* each function returns a pair of (valuem, feasible) where feasible is a boolean indicating if the value is feasible or not