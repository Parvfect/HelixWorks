

import numpy as np

def remove_from_array(vals, current_value):
        """ Removes current value from vals"""

        new_vals = []
        for i in range(len(vals)):
            if (current_value == vals[i]).all():
                return [*new_vals, *vals[i:]]
            new_vals.append(vals[i])
        return new_vals 



index = 3
t = np.random.rand(5)
s = t[index]

print(remove_from_array(t, s) == [*t[:index], *t[index:]])