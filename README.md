# Diagfig
A reviewer asked you for a b/w & colourblind friendly figure? `diagfig` can be used for a very simple visual checking.

Other solutions exist for checking in a palette or an image is colourblind friendly, but I needed a tool that I could more easily plug on `matplotlib` that would check in one line a figure. The code is vaslty inspired by other solutions but avoid too many dependencies.

## Example of use
The following example provide a few possibilities of use of `diagfig`. Similar to example_1.py.

```python
import numpy as np
import matplotlib.pyplot as plt
import diagfig

# creating a dummy figure in matplotlib
fig, ax = plt.subplots()
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color="tab:blue", alpha=.5)
ax.plot(x, y, color="tab:red", lw=2, ls="--")

# diagnosing the figure
fig_diag = diagfig.diagnose_figure(fig)
```

A decorator `@diag_it` is also available in `Diagfig`. Similar to example_2.py.
```python
import numpy as np
import matplotlib.pyplot as plt
from diagfig import diag_it

# creating a dummy figure in matplotlib
@diag_it
def simple_plot():
    fig, ax = plt.subplots()
    x = np.arange(10)
    y_low = x * .5
    y_high = x * 2
    y = x.copy()
    ax.fill_between(x, y_low, y_high, color="tab:blue", alpha=.5)
    ax.plot(x, y, color="tab:red", lw=2, ls="--")
    return fig

# diagnosing the figure
simple_plot()
```

It is also possible to initiate `matplotlib` figures with the custom subclass `FigureDiag`

```python
import numpy as np
import matplotlib.pyplot as plt
from diagfig import FigureDiag

# creating a dummy figure in matplotlib using the custom subclass of matplotli.figure.Figure
fig, ax = plt.subplots(FigureClass=FigureDiag)
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color="tab:blue", alpha=.5)
ax.plot(x, y, color="tab:red", lw=2, ls="--")

# diagnosing the figure
diaged_fig = fig.diag()
```
### Output generated:

![ExampleUse](example/example_use.png)


## Development
Feel free to modify, remove and/or add functionality, but make sure to set up automatic tests to show the working of your 
adaptations. 

### Type Hints
Use type hints ([Python Documentation](https://docs.python.org/3/library/typing.html)) whenever possible, 
**especially in method signatures**. 
A type hint gives information about the type of a variables. Consider the following example:

```python
def is_even(a: int) -> bool:

    if a % 2 == 0:
        return True
    else:
        return False
```

The function _is_even_ requires an integer as argument and will return a boolean. If the function returned a string,
this would be considered a type error and the program would not be accepted. 

Adding type information, restricts which kinds of values can be stored by variables, or be returned by methods. This
 makes programs easier to understand and helps to detect errors before actually executing a piece of code.

When creating new classes, consider this example on how to set up 
getters/setters using properties ([Python Documentation](https://docs.python.org/3/library/functions.html#property)):
```python
# Source: https://github.com/python/mypy/issues/220
class C:
    def __init__(self) -> None:
        self._x = 0

    @property
    def x(self) -> int:
        return self._x

    @x.setter
    def x(self, value: int) -> None:
        self._x = value

    @x.deleter
    def x(self) -> None:
        del self._x

c = C()
c.x = 2 * c.x
c.x = ''   # Error!
```

Type hints will be checked automatically on GitHub and errors will be reported. Locally, type hints can be checked using 
the _mypy_ tool ([Documentation](http://mypy-lang.org/)):

```python
mypy source_file.py
```

### Automatic Tests
To create a test that is automatically executed, create a new class called "Test" in a file starting with "test_" and 
save it in the folder _tests/_. The class should have a similar structure as the following example:

```python
# Copied from: https://docs.python.org/3/library/unittest.html

import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
```

#### Summary:
* File is stored in folder _tests/_
* File starts with "test_"
* Class starts with "Test"
* Class inherits from unittest.TestCase
* Test methods start with "test_"

Check the [Python Documentation](https://docs.python.org/3/library/unittest.html) for more details on how to create 
unittests. 