import numpy as np
import matplotlib.pyplot as plt

def plot_error(original_history, x, file_name, sub_plot=False, ax=None, label=None, mode='normal'):
    history = np.copy(original_history)
    history = np.array(history)
    history = history - x
    history = np.linalg.norm(history, axis=1)

    if not sub_plot:
        if "log" in mode:
            plt.plot(np.log(history))
        else:
            plt.plot(history)
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.savefig('./imgs/' + file_name + '.png')
        plt.show()
    else:
        if "log" in mode:
            ax.plot(np.log(history), label=label)
        else:
            ax.plot(history, label=label)


def plot_direction(original_history, x, file_name, sub_plot=False, ax=None):
    history = np.copy(original_history)
    i = 1
    dff_history = []
    while i < len(history):
        step_dff = history[i] - history[i - 1]
        step_norm = np.linalg.norm(step_dff)

        opt_dff = x - history[i - 1]
        opt_norm = np.linalg.norm(opt_dff)

        dff = np.dot(np.transpose(step_dff), opt_dff) / (step_norm * opt_norm)
        dff_history.append(float(dff))
        i += 1
    if not sub_plot:
        plt.plot(dff_history)
        plt.xlabel('k')
        plt.ylabel('direction difference')
        plt.savefig('./imgs/' + file_name + '.png')
        plt.show()
    else:
        ax.plot(dff_history[:200], label=file_name)




def plot_exp(original_history, x, file_name, sub_plot =False, ax=None):
    history = np.copy(original_history)
    history = history - x
    history = np.linalg.norm(history, axis=1)

    # compute the exponential function
    if plot_exp:
        c = float(history[0])
        mid = int(len(history) / 2)
        u = float(history[mid + 1] / history[mid])
        exp = [c]
        for i in range(len(history)):
            c = c * u
            exp.append(c)
        exp = np.array(exp)

        # plot the error and the exponential function to see their connections
        fig, ax = plt.subplots()
        ax.plot(history, label='error')
        ax.plot(exp, label='exp')

        ax.set_xlabel('iteration')
        ax.set_ylabel('error')
        ax.legend()

        plt.savefig('./imgs/' + file_name + '.png')
        plt.show()