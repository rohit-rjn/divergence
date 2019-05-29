from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.layouts import column
#output_notebook()

#Plotting the extermas
def plot_extrema(df,values, mintab,maxtab,y_axis):
    df=df.fillna(0)
    # create a new plot with a title and axis labels
    p = figure(title=y_axis, x_axis_label='Index', y_axis_label=y_axis, plot_width=1000, plot_height=400)

    # add a line renderer with legend and line thickness
    p.line(df.index,values, legend='value', line_width=0.6)
    p.scatter(np.array(maxtab)[:,0],np.array(maxtab)[:,1],
          fill_color='red', fill_alpha=1,
          line_color=None)
    p.scatter(np.array(mintab)[:,0],np.array(mintab)[:,1],
          fill_color='green', fill_alpha=1,
          line_color=None)
    show(p)