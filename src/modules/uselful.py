import numpy as np
import uncertainties as un
import matplotlib.pyplot as plt
import pandas as pd


def un_separator(ufloat):
    if isinstance(ufloat, (list, np.ndarray, pd.DataFrame)):
        nominal = np.array([i.n for i in ufloat])
        uncertan = np.array([i.s for i in ufloat])
    elif isinstance(ufloat, un.UFloat):
        nominal = ufloat.n
        uncertan = ufloat.s
    return [nominal, uncertan]

def un_merger(nominal, uncertain):
    ufloat_array = []
    for i, j in zip(nominal,uncertain):
        ufloat_array.append(un.ufloat(i,j))
    return np.array(ufloat_array)



def const_un(nominal: list, uncertainty: float):
    ufloat_array = [un.ufloat(i,uncertainty) for i in nominal]
    return ufloat_array



def un_plot(fig,ax, x:list,y:list,x_label = "$x$", y_label ="$y$",title=" ",label=" " ,format= "b+", cap = 5 , figsize=(12,8), show_error= "errorbar", save = 0):
    y, dy = un_separator(y)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_error == "errorbar":
        x , dx = un_separator(x)
        ax.errorbar(x,y,yerr= dy, xerr = dx, fmt=format , capsize=cap, label=label)
    if show_error == "fill_between":
        ax.fill_between(x,y-dy,y+dy,alpha=0.3)
        ax.plot(x,y,format,label=label)
    if type(save) == str:
        fig.savefig(save,dpi="figure")



def un_plot_simple(x:list,y:list,format = "b+", show_error = "errorbar", label = "", cap =  5, unlabel=""):
    if show_error == "errorbar":
        y, dy = un_separator(y)
        x , dx = un_separator(x)
        plt.errorbar(x,y,yerr= dy, xerr = dx, fmt=format , capsize=cap, label=label)
    if show_error == "fill_between":
        x, dx = un_separator(x)
        y, dy = un_separator(y)
        plt.fill_between(x,y-dy,y+dy,alpha=0.3,color=format,label=unlabel)
        plt.plot(x,y,format,label=label)
  
