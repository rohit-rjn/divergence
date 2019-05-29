from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
from bokeh.layouts import column
#output_notebook()

def plot_divergence(df, maxtab_close, mintab_close,maxtab_macd, mintab_macd, title):
    df=df.fillna(0)
    
    p1 = figure(title=title, x_axis_label="index",x_axis_type='datetime', y_axis_label='Price', plot_width=1000, plot_height=400)
    p1.line(df['date'],df['kf'], legend="KF", line_width=0.6, line_color='black')
    p1.scatter(df.where(df['regular_signal'] == 1)['date'], df.where(df['regular_signal'] == 1)['ema'],
          fill_color='green', fill_alpha=1,legend='bull',
          line_color=None)
    p1.scatter(df.where(df['hidden_signal'] == 2)['date'], df.where(df['hidden_signal'] == 2)['ema'],
          fill_color='blue', fill_alpha=1,legend='hidden bull',
          line_color=None)
    p1.scatter(df.where(df['regular_signal'] == 3)['date'], df.where(df['regular_signal'] == 3)['ema'],
          fill_color='#A53434', fill_alpha=1,legend='bear',
          line_color = None)
    p1.scatter(df.where(df['hidden_signal'] == 4)['date'], df.where(df['hidden_signal'] == 4)['ema'],
          fill_color='#FFD900', fill_alpha=1,legend='hidden bear',                                                                                                                             
          line_color=None)
    
    p4 = figure(title='Complete Divergence', x_axis_label="index",x_axis_type='datetime', y_axis_label='Price', plot_width=1000, plot_height=400)
    p4.line(df['date'],df['ema'], legend="EMA-5-Close", line_width=0.6, line_color='black')
    p4.scatter(df.where(df['signal_complete'] == 1)['date'], df.where(df['signal_complete'] == 1)['ema'],
          fill_color='green', fill_alpha=1,legend='bull',
          line_color=None)
    #p1.scatter(df.where(df['hidden_signal'] == 2).index, df.where(df['hidden_signal'] == 2)['ema'],
    #      fill_color='blue', fill_alpha=1,legend='hidden bull',
    #      line_color=None)
    p4.scatter(df.where(df['signal_complete'] == 3)['date'], df.where(df['signal_complete'] == 3)['ema'],
          fill_color='#A53434', fill_alpha=1,legend='bear',
          line_color=None)
    #p1.scatter(df.where(df['hidden_signal'] == 4).index, df.where(df['hidden_signal'] == 4)['ema'],
    #      fill_color='#FFD900', fill_alpha=1,legend='hidden bear',                                                                                                                             
    #      line_color=None)
    
# create a new plot with a title and axis labels  
    p2 = figure(title=title, x_axis_label='Index', y_axis_label='MACD Hist', plot_width=1000, plot_height=200)

    # add a line renderer with legend and line thickness
    p2.line(df.index,df['macdhist'], legend="MACD Histogram", line_width=0.6,line_color='black')
    p2.scatter(np.array(maxtab_macd)[:,0],np.array(maxtab_macd)[:,1],
          fill_color='red', fill_alpha=1,
          line_color=None)
    p2.scatter(np.array(mintab_macd)[:,0],np.array(mintab_macd)[:,1],
          fill_color='green', fill_alpha=1,
          line_color=None)
    
    p3 = figure(title="MACD (12, 26, close, 9)", x_axis_label='index',x_axis_type='datetime', plot_width=1000, plot_height=400, toolbar_location=None, x_range=p1.x_range)
    p3.quad(top=df.macdhist[1:], bottom=0, left=df.date[:-1], right=df.date[1:], alpha=0.5)

    #output_file(title+".html")
    show(column(p1, p2, p3,p4))
    #show(p3)