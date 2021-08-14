# Copyright (c) 2021 Koichi Sakata

# close_all()
# clear_all()


def close_all():
    from matplotlib import pyplot as plt
    plt.close('all')
    print('All figures were closed.')


def clear_all():
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
    print('All parameters were cleared.')