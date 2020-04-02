# convTest_conf.py
'''
WIP
'''

# Example: made with numpy logspace from 1e0 to 1e-3.
du_list = [ 1.,
            0.46415888,
            0.21544347,
            0.1,
            0.04641589,
            0.02154435,
            0.01,
            0.00464159,
            0.00215443,
            0.001 ]
du_type = 'dt'

# Example: folder names in ../data with data for beef objects that have already been produced.
calc_folders = [ 'f0',
                 'f1',
                 'f2',
                 'f3',
                 'f4',
                 'f5',
                 'f6',
                 'f7',
                 'f8',
                 'f9' ]

# Example: folder name in ../data with interpolation of analytic solution, having the same shape as the other beefs.
exact_folder = [ 'something' ]

# Example: the time at which the comparison is to be evaluated.
t = 2.25

# Example: what we are measuring the convergence of. Must be either 'T' or 'C'.
quantity = 'T'

# Example - This dataset is typically small and may safely be stored in memory.
data_savefile = '/data/conv_data1.csv'

# Example
plot_savefile = '/data/plots/conv_plot1.pdf'

# TODO: Make completely consistent with convergence_test.py
plot_data = {
    'x_label':     r'dt',
    'y_label':     'Error',
    'title':       f'Error in measuring {quantity} at time {t} s.',
    
    # data_savefile could probably be split into data_savefile_T and data_savefile_C
    'data_savefile' : data_savefile,
    'plot_savefile':    plot_savefile,
    
    'line_colour': 'red',
    'line_width':  4,
    # ...?
}

convTest_conf = {
    'du_list':      du_list,
    'du_type':      du_type,
    'calc_folders': calc_folders,
    'exact_folder': exact_folder,
    't':            t,
    'quantity':     quantity,
    'plot_data':    plot_data
}
