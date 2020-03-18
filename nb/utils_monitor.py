import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from eden.display import draw_graph, draw_graph_set, map_labels_to_colors
import time

def display_graph_list(graphs, oracle_func, score_estimator, draw_graphs=None):    
    true_scores = np.array([oracle_func(g) for g in graphs])
    oracle_ids = np.argsort(-true_scores)
    pred_scores = np.array([score_estimator.predict([g])[0] for g in graphs])
    pred_ids = np.argsort(-pred_scores)
    
    #oracle
    titles = []
    gs = []
    n=2
    for id in oracle_ids[:n+1]:
        true_score = true_scores[id]
        pred_score = pred_scores[id]
        titles.append('true:%.3f pred:%.3f  '%(true_score, pred_score))
        gs.append(graphs[id])
    
    #preds
    for id in pred_ids[:n]:
        true_score = true_scores[id]
        pred_score = pred_scores[id]
        titles.append('true:%.3f pred:%.3f  '%(true_score, pred_score))
        gs.append(graphs[id])
        
    draw_graphs(gs, titles=titles, n_graphs_per_line=2*n+1)

def smooth(x,y, sigma_fact=7):
    sigma = (max(x)-min(x))/sigma_fact
    xnew = np.linspace(min(x), max(x), 200)
    gy = gaussian_filter1d(y, sigma)
    f = interpolate.InterpolatedUnivariateSpline(x, gy)
    ynew = f(xnew)
    return xnew, ynew 
    
def plot_status(estimated_mean_and_std_target, current_best, scores_list, num_oracle_queries, sigma_fact=7):
    fig = plt.figure(figsize=(17,5))
    ax1 = fig.add_subplot(1, 1, 1)

    if len(estimated_mean_and_std_target) == len(scores_list):
        # target with variance
        estimated_mean_and_std_target_array = np.array(estimated_mean_and_std_target)
        target_means, target_stds = estimated_mean_and_std_target_array.T

        ax1.fill_between(range(len(estimated_mean_and_std_target)), target_means+target_stds, target_means-target_stds, alpha=.1, color='steelblue')
        ax1.fill_between(range(len(estimated_mean_and_std_target)), target_means+target_stds/10, target_means-target_stds/10, alpha=.1, color='steelblue')
        ax1.fill_between(range(len(estimated_mean_and_std_target)), target_means+target_stds/100, target_means-target_stds/100, alpha=.1, color='steelblue')
        ax1.plot(target_means, linestyle='dashed')
        xx, m = smooth(range(len(target_means)),target_means,sigma_fact)
        ax1.plot(xx, m, lw=5, color='steelblue', label='true target graph scored by predictor')
    
    # median and violinplot
    #plt.violinplot(scores_list, range(len(scores_list)), points=60, widths=0.7, showmeans=True, showextrema=True, showmedians=True, bw_method=0.3)
    medians = [np.median(scores) for scores in scores_list]
    ax1.plot(medians, color='darkorange', lw=1, linestyle='dotted')
    xx, m = smooth(range(len(medians)), medians, sigma_fact)
    ax1.plot(xx, m, lw=3, linestyle='dashed', color='darkorange', label='median of generated graphs scored by oracle')

    #current best
    ax1.plot(current_best, color='darkorange', linestyle='dashed')
    xx, m = smooth(range(len(current_best)), current_best,sigma_fact)
    ax1.plot(xx, m, lw=5, color='darkorange', label='current opt graph scored by oracle')
    ax1.legend()
    y_low = max(0,min(min(medians), min(current_best)))
    y_up = min(1,max(max(medians), max(current_best)))
    ax1.set_ylim(y_low,y_up)
    ax1.set_xlabel('num iteration')
    ax1.set_ylabel('score')
    ax1.grid()
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(num_oracle_queries, linestyle='dotted', color='gray', alpha=.5, label='# queries to oracle')
    ax2.set_ylabel('# queries')
    fig.tight_layout()
    plt.show()
    


def make_monitor(target_graph=None, oracle_func=None, draw_graphs=None, draw_history=None, show_step=1):
    history = []
    estimated_mean_and_std_target=[]
    current_best=[]
    scores_list=[]  
    duration = []
    num_oracle_queries = []

    def monitor(i, graphs, all_graphs, score_estimator):
        num_oracle_queries.append(len(all_graphs))
        history.extend(graphs[:])
        if target_graph is not None:
            mu, sigma = score_estimator.predict([target_graph]), score_estimator.predict_uncertainty([target_graph])
            estimated_mean_and_std_target.append((mu[0],sigma[0]))
        
        true_scores = [oracle_func(g) for g in graphs]
        pred_scores = score_estimator.predict(graphs)
            
        scores_list.append(true_scores)
        best_score = max(true_scores)
        best_graph = graphs[np.argmax(true_scores)]
        print('< %.3f > best score in %d newly generated instances'%(best_score, len(graphs)))
        score, explanation = oracle_func(best_graph, explain=True)
        explanation_str = ' '.join(['%s:%.3f'%(key,val) for key,val in sorted(explanation.items())])
        print('    score decomposition: %.3f = geom. mean of: %s'%(score, explanation_str))
        current_best.append(best_score)
        duration.append(time.clock())
        if i>0 and (show_step==1 or i%show_step==0):
            if len(scores_list)>5:
                plot_status(estimated_mean_and_std_target,current_best, scores_list, num_oracle_queries, 5)
            if len(duration)>2: print('Correlation coefficient true score vs predicted score: %.3f  runtime:%.1f mins' % (np.corrcoef(true_scores,pred_scores)[0,1], (duration[-1]-duration[-2])/60))       
            if target_graph is not None:
                display_graph_list(graphs+[target_graph], oracle_func, score_estimator, draw_graphs=draw_graphs)
            else:
                display_graph_list(graphs, oracle_func, score_estimator, draw_graphs=draw_graphs)
            print('Evolution of current best proposal')
            draw_history(graphs, oracle_func)
            
    return monitor