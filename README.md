# Diagfig
A reviewer asked you for a b/w & colorblind friendly figure? `diagfig` can be used for a very simple visual checking.

### Motivation and disclaimer

The original idea for `diagfig` was a demo package for getting used to packaging in python and that would have a few helpers to make sure a figure is ready for scientific publication. Additional functionalities may be added in the future, but the current version support colorblindness simulation and low dpi simulation.

This is by no mean my field of research, see references and docstrings for further information.

Other solutions exist for checking in a palette or an image is colorblind friendly, but I needed a tool that I could more easily plug on `matplotlib` that would check in one line a figure. The code is vastly inspired by other solutions but avoid too many dependencies.

## Install the package

```bash
pip install diagfig
```

## Examples of use
There are a few options for using `diagfig`. They are meant to help you to diagnose if a figure require color / quality adjustment before publication.
It can be used to simulate colorblindness on your figure as well as black and white.
You can also play with the dpi to combine effect of colorbliness and poor export quality.
The main options are:
- `diagfig.diagnose_figure`: a function taking a figure as argument and returning a diagnosed figure.
- `diagfig.diag_it`: a decorator automatically generating a diagnosed figure.
- `diagfig.FigureDiag`: a subclass of `matpotlib.figure.Figure` with an additional `diag` method returning a "diagnosed" figure.

Use of the main function `diagnose_figure`:
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

Use of the decorator `@diag_it`:
```python
import numpy as np
import matplotlib.pyplot as plt
from diagfig import diag_it

# creating a dummy figure in matplotlib
@diag_it()
def simple_plot()-> plt.Figure:
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

Use of the `matplotlib.figure.Figure` custom subclass `FigureDiag`:
```python
import numpy as np
import matplotlib.pyplot as plt
from diagfig import FigureDiag

# creating a dummy figure in matplotlib using the custom subclass of matplotlib.figure.Figure
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

## References
1. Lee, J., & Santos, W. P. dos. (2011). An Adaptive Fuzzy-Based System to Simulate, Quantify and Compensate Color Blindness. Integrated Computer-Aided Engineering, 18(1), 29–40. https://doi.org/10.3233/ICA-2011-0356

2. Lindbloom, B. (2017. April 06). RGB Working Space Information. Retrieved 14 August 2024, from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html

3. LMS color space. (2024). In Wikipedia. Retrieved 14 August 2024, from https://en.wikipedia.org/w/index.php?title=LMS_color_space&oldid=1233434413#Hunt.2C_RLAB

4. Viénot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps for checking the legibility of displays by dichromats. Color Research & Application, 24(4), 243–252. https://doi.org/10.1002/(SICI)1520-6378(199908)24:4<243::AID-COL5>3.0.CO;2-3

5. Ruminski, J., Bajorek, M., Ruminska, J., Wtorek, J., & Bujnowski, A. (2012). Computerized Color Processing for Dichromats. In Z. S. Hippe, J. L. Kulikowski, & T. Mroczek (Eds.), Human – Computer Systems Interaction: Backgrounds and Applications 2: Part 1 (pp. 453–470). Springer. https://doi.org/10.1007/978-3-642-23187-2_29

6. Schmitz, J. (2016, August 28). Color Blindness Simulation Research. Ixora.Io. Retrieved 14 August 2024, from https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

7. Thakkar, S. (2024). Tsarjak/Simulate-Correct-ColorBlindness [Python]. https://github.com/tsarjak/Simulate-Correct-ColorBlindness (Original work published 2017)