import BeefSimulator as BS

if __name__ == '__main__':
    h = 0.25
    dt = 0.1
    dims = [[0,5], [0,4], [0,3], [0, 10]]
    bs = BS.BeefSimulator(dims, h, 1, dt = dt)
    bs.plotter.show_heat_map('../data/test_temp_dist.npy', 5, x = 4.5)