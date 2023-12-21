

import numpy as np

def remove_from_array(vals, current_value):
        """ Removes current value from vals"""

        new_vals = []
        for i in range(len(vals)):
            if np.array_equal(vals[i], current_value):
                continue
            new_vals.append(vals[i])
        return new_vals 



index = 4
t = np.random.rand(5)
s = t[index]
y = t.copy()

print(t)
print(s)
print(y)
print(remove_from_array(t, s))