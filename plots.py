from matplotlib import pyplot as plt
import json

if __name__ == '__main__':
    history = json.load(open('lr_finder.json'))
    lrs = history['lr'][:]
    losses = history['loss'][:]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid()
    ax.grid(which='minor', linestyle='--')
    plt.show()