import logging
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn import metrics


# Use scikitplot.metrics.plot_roc - nice because has adds other blended ROC curves in it.
def plot_roc(y_true, y_probas, **kwargs):
    import scikitplot.metrics
    #Reorder y_probas columns to match order of categories in y_true
    #plot_roc() expects alphabetical, but labels are probably ordered categories in a different order
    new_column_order = np.argsort(y_true.cat.categories)
    y_probas_new = y_probas[:, new_column_order]
    return scikitplot.metrics.plot_roc(y_true, y_probas_new, **kwargs)



def limits(stat):
    x = {
        'BRIER'   : [ 0   ,0.15],
        'CSI'     : [ 0   ,1],
        'FSS'     : [ 0   ,1],
        'GSS'     : [ 0   ,1],
        'BAGSS'   : [ 0   ,1], # bias adjusted GSS
        'HK'      : [-0.2 ,1],
        'HSS'     : [-0.2 ,1],
        'BSS_SMPL': [-1   ,1],
        'ROC_AUC' : [ 0.45,1],
        }
    if stat not in x.keys():
        return [0,1]
    return x[stat]

def bss(obs, fcst):
    bs = np.mean((fcst-obs)**2)
    climo = np.mean((obs - np.mean(obs))**2)
    return 1.0 - bs/climo

def reliability_diagram(ax, obs, fcst, base_rate=None, label="", n_bins=10, debug=False, **kwargs):
    if obs is None or fcst is None:
        # placeholder
        p_list = ax.plot( [0], [0], "s-", label=label, **kwargs)
    else:
        # allow lists
        obs = np.array(obs)
        fcst = np.array(fcst)
        # calibration curve
        true_prob, fcst_prob = calibration_curve(obs, fcst, n_bins=n_bins)
        bss_val = bss(obs, fcst)
        if base_rate is None:
            base_rate = obs.mean() # base rate
        p_list = ax.plot( fcst_prob, true_prob, "s-", label="%s  bss:%1.4f" % (label, bss_val), **kwargs)
        for x, f in zip(fcst_prob, true_prob):
            if np.isnan(f): continue # avoid TypeError: ufunc 'isnan' not supported...
            # label reliability points
            ax.annotate("%1.3f" % f, xy=(x,f), xycoords=('data', 'data'), 
                xytext = (0,1), textcoords='offset points', va='bottom', ha='center',
                fontsize='xx-small')
    p = p_list[0]
    one2oneline_label = "Perfectly calibrated"
    # If it is not a child already add perfectly calibrated line
    has_one2oneline = one2oneline_label in [x.get_label() for x in ax.get_lines()]
    if not has_one2oneline:
        one2oneline = ax.plot([0, 1], [0, 1], "k:", alpha=0.7, label=one2oneline_label)
    noskill_line = ax.plot([0, 1], [base_rate/2, (1+base_rate)/2], color=p.get_color(), linewidth=0.3, alpha=0.7)
    baserateline          = ax.axhline(y = base_rate, color=p.get_color(), label=f"base rate {base_rate:.4f}", linewidth=0.5, linestyle="dashed", dashes=(9,9))
    baserateline_vertical = ax.axvline(x = base_rate, color=p.get_color(), linewidth=0.5, linestyle="dashed", dashes=(9,9))

    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left", fontsize="xx-small")
    ax.grid(lw=0.5, alpha=0.5)
    ax.set_xlim((0,1))

    return p_list
   
def count_histogram(ax, fcst, label="", n_bins=10, count_label=True, debug=False):
    ax.set_xlabel("Forecasted probability")
    ax.set_ylabel("Count")
    ax.grid(lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    if fcst is None: return None
    # Histogram of counts
    counts, bins, patches = ax.hist(fcst, bins=n_bins, label=label, histtype='step', lw=2, alpha=1, log=True)
    if count_label:
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            # label raw counts
            ax.annotate(str(int(count)), xy=(x,count), xycoords=('data', 'data'),
                xytext = (0,-1), textcoords='offset points', va='top', ha='center', fontsize='xx-small')
    return counts, bins, patches

def ROC_curve(ax, obs, fcst, label="", sep=0.1, plabel=True, fill=False):
    auc = None
    if obs.all() or (obs == False).all():
        logging.info("obs are all True or all False. ROC AUC score not defined")
        r = ax.plot([0],[0], marker="+", linestyle="solid", label=label)
    elif obs is None or fcst is None:
        # placeholders
        r = ax.plot([0],[0], marker="+", linestyle="solid", label=label)
    else:
        # ROC auc with threshold labels separated by sep
        auc = metrics.roc_auc_score(obs, fcst)
        logging.debug(f"auc {auc}")
        pofd, pody, thresholds = metrics.roc_curve(obs, fcst)
        r = ax.plot(pofd, pody, marker="+", markersize=1/np.log10(len(pofd)), linestyle="solid", label="%s  auc:%1.4f" % (label, auc))
        if fill:
            auc = ax.fill_between(pofd, pody, alpha=0.2)
        if plabel:
            old_x, old_y = 0., 0.
            for x, y, s in zip(pofd, pody, thresholds):
                if ((x-old_x)**2+(y-old_y)**2.)**0.5 > sep:
                    # label thresholds on ROC curve
                    ax.annotate("%1.3f" % s, xy=(x,y), xycoords=('data', 'data'),
                            xytext=(0,1), textcoords='offset points', va='baseline', ha='left',
                            fontsize = 'xx-small')
                    old_x, old_y = x, y
                else:
                    logging.debug(f"statisticplot.ROC_curve(): tossing {x},{y},{s} annotation. Too close to last label.")
    ax.set_title("ROC curve")
    no_skill_label = "no skill:0.5"
    # If it is not a child already add perfectly calibrated line
    has_no_skill_line = no_skill_label in [x.get_label() for x in ax.get_lines()]
    if not has_no_skill_line:
        no_skill_line = ax.plot([0, 1], [0, 1], "k:", alpha=0.7, label=no_skill_label)
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xlabel("Prob of false detection")
    ax.set_ylabel("Prob of true detection")
    ax.grid(lw=0.5, alpha=0.5)
    ax.legend(loc="lower right", fontsize="xx-small")
    return r, auc
