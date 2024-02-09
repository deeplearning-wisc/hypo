import scipy.io as io        
import numpy as np
import ot
from itertools import combinations
import torch
from geomloss import SamplesLoss
import matplotlib.pyplot as plt


def compute_wasserstein_distance(X1, X2, metric = 'euclidean', reg=1e-2):
    '''
    numerically unstable with OT library
    '''
    # Calculate the cost matrix, which is the pairwise Euclidean distance
    cost_matrix = ot.dist(X1, X2, metric=metric)

    # Compute the empirical distribution weights, assuming all samples have equal probability
    n1_samples, n2_samples  = X1.shape[0], X2.shape[0]
    weights_x1 = np.ones(n1_samples) / n2_samples
    weights_x2 = np.ones(n2_samples) / n2_samples

    # Compute the Wasserstein distance using the Sinkhorn algorithm
    wasserstein_distance = ot.sinkhorn2(weights_x1, weights_x2, cost_matrix, reg=reg, stopThr=1e-8)
    
    return wasserstein_distance


def compute_wasserstein_distance2(X1, X2, p=2, reg=0.01):
    # https://www.kernel-operations.io/geomloss/api/pytorch-api.html
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X1_torch = torch.tensor(X1, dtype=torch.float32, device=device)
    X2_torch = torch.tensor(X2, dtype=torch.float32, device=device)

    # Initialize the SamplesLoss function with the Wasserstein distance (p=2)
    # loss="sinkhorn"： (Un-biased) Sinkhorn divergence, which interpolates between Wasserstein (blur=0) and kernel (blur= 
    #) distances.
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=reg)

    # Compute the Wasserstein distance
    wd = loss(X1_torch, X2_torch)

    # Convert the result to a scalar value
    wd_scalar = wd.item()
    return wd_scalar


def plot_heatmap(data, save_name):
    # Find the maximum values of y and e
    max_y = max(data.keys())
    # max_e = max(max(pair for pair in data[y].keys()) for y in data.keys()) #e.g. (2,3)

    unique_pairs = list(data[0].keys())
    print(unique_pairs)
    # Initialize an empty matrix to store the data
    matrix = np.zeros((max_y + 1, len(unique_pairs)))

    for y, pairs in data.items():
        for (e1, e2), value in pairs.items():
            pair_index = unique_pairs.index((e1, e2))
            matrix[y, pair_index] = value

    # Plot the heatmap using plt.imshow
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    ax.set_xlabel("(e1, e2)", fontsize=14)
    ax.set_xticks(range(len(unique_pairs)), unique_pairs, rotation=45, fontsize=12)
    ax.set_ylabel("y", fontsize=14)
    ax.tick_params(axis='y', labelsize=12) # y tickle size

    # Add numbers to each block of the heatmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w", fontsize=10)

    plt.tight_layout()
    fig.colorbar(im, ax=ax, label="Wasserstein distance (sinkhorn divergence)")
    plt.title("")
    plt.tight_layout()
    plt.savefig(f'plots/{save_name}.png', dpi = 300,bbox_inches='tight')


if __name__ == '__main__': 
    # classwise
    normalize = True 
    for loss in ['ce', 'cider']:
        for normalize in ['no_norm', 'norm']:
            save_name = f'feature_y_e_{loss}_penultimate_{normalize}'
            res = {}
            feature_y_e=io.loadmat(f'features/{save_name}.mat')
            all_features = feature_y_e['feature']
            all_y = feature_y_e['y'].squeeze()
            all_e = feature_y_e['e'].squeeze()
            unique_y = np.unique(all_y)
            unique_e = np.unique(all_e)
            unique_e_pairs = list(combinations(unique_e, 2))
            for y in unique_y:
                res[y] = {}
                for (e1, e2) in unique_e_pairs:
                    feat_1 = all_features[(all_y == y) & (all_e == e1)] # element-wise & (bitwise AND) operator to combine the boolean arrays; Remember to use parentheses around each condition since the & operator has higher precedence.
                    feat_2 = all_features[(all_y==y) & (all_e == e2)]
                    # print(feat_1.shape, feat_2.shape)
                    # wd = compute_wasserstein_distance2(feat_1.astype('float64'), feat_2.astype('float64'))
                    wd = compute_wasserstein_distance2(feat_1, feat_2)
                    print(f'y = {y}, estimated W-dist for ({e1}, {e2}) is {wd}')
                    res[y][(e1,e2)] = wd
            
            print(res)
            # debug test
            # data = {0: {(0, 1): 0.15530866384506226, (0, 2): 0.21642754971981049, (0, 3): 0.23963293433189392, (1, 2): 0.20890945196151733, (1, 3): 0.26223963499069214, (2, 3): 0.2195998877286911}, 1: {(0, 1): 0.14590714871883392, (0, 2): 0.23728705942630768, (0, 3): 0.28713685274124146, (1, 2): 0.23684373497962952, (1, 3): 0.3055537939071655, (2, 3): 0.20455986261367798}, 2: {(0, 1): 0.15946874022483826, (0, 2): 0.1995120495557785, (0, 3): 0.1753748506307602, (1, 2): 0.20574313402175903, (1, 3): 0.2013503611087799, (2, 3): 0.17896652221679688}, 3: {(0, 1): 0.13759663701057434, (0, 2): 0.18846842646598816, (0, 3): 0.1469249725341797, (1, 2): 0.15395350754261017, (1, 3): 0.093973807990551, (2, 3): 0.12456567585468292}, 4: {(0, 1): 0.16651654243469238, (0, 2): 0.23089328408241272, (0, 3): 0.2536894679069519, (1, 2): 0.22801926732063293, (1, 3): 0.2667630910873413, (2, 3): 0.23195475339889526}, 5: {(0, 1): 0.10270039737224579, (0, 2): 0.20019754767417908, (0, 3): 0.16466757655143738, (1, 2): 0.1986047923564911, (1, 3): 0.16104549169540405, (2, 3): 0.17241033911705017}, 6: {(0, 1): 0.2291284203529358, (0, 2): 0.24648775160312653, (0, 3): 0.24859079718589783, (1, 2): 0.2752886712551117, (1, 3): 0.3062085807323456, (2, 3): 0.25627970695495605}}
            plot_heatmap(res,save_name)