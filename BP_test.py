import BeefSimulator as BS

# Run testdata_gen.py first!
if __name__ == '__main__':
    h = 0.25
    dt = 0.1
    dims = [[0,25], [0,37.5], [0,50.0], [0, 10]]
    bs = BS.BeefSimulator(dims, h, 1, dt = dt)
    bs.plotter.show_heat_map('../data/test_temp_dist.npy', 0, x = 12.5)
    bs.plotter.show_heat_map('../data/test_temp_dist.npy', 1, z = 25)