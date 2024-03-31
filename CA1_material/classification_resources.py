import sys, csv, datetime
import numpy
import matplotlib.pyplot as plt
import pdb

# datetime can be used for tracking running time
# tic = datetime.datetime.now()
# [running some function that takes time]
# tac = datetime.datetime.now()
# print("Function running time: %s" % (tac-tic))
# print("The function took %s seconds to complete" % (tac-tic).total_seconds())

# For repeatability of experiments using randomness
class RandomNumGen:

    generator = None
    seed = None

    @classmethod
    def set_seed(tcl, seed=None):
        tcl.seed = seed
        if seed is not None:
            tcl.generator = numpy.random.Generator(numpy.random.MT19937(numpy.random.SeedSequence(seed)))
        else:
            tcl.generator = None

    @classmethod
    def get_gen(tcl):
        if tcl.generator is None:
            return numpy.random.default_rng()
        return tcl.generator


# Load a CSV file
def load_csv(filename, last_column_str=False, normalize=False, as_int=False):
    dataset = list()
    head = None
    classes = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for ri, row in enumerate(csv_reader):
            if not row:
                continue
            if ri == 0:
                head = row
            else:
                rr = [r.strip() for r in row]
                if last_column_str:
                    if rr[-1] not in classes:
                        classes[rr[-1]] = len(classes)
                    rr[-1] = classes[rr[-1]]
                dataset.append([float(r) for r in rr])
    dataset = numpy.array(dataset)
    if not last_column_str and len(numpy.unique(dataset[:,-1])) <= 10:
        classes = dict([("%s" % v, v) for v in numpy.unique(dataset[:,-1])])
    if normalize:
        dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    return dataset, head, classes

# Find the min and max values for each column
def dataset_minmax(dataset):
    return numpy.vstack([numpy.min(dataset, axis=0), numpy.max(dataset, axis=0)])

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax=None):
    if minmax is None:
        minmax = dataset_minmax(dataset)
    return (dataset - numpy.tile(minmax[0, :], (dataset.shape[0], 1))) / numpy.tile(minmax[1, :]-minmax[0, :], (dataset.shape[0], 1))

# Sample k random points from the domain
def sample_domain(k, minmax=None, dataset=None):
    if dataset is not None:
        minmax = dataset_minmax(dataset)
    if minmax is None:
        return RandomNumGen.get_gen().random(k)
    d = RandomNumGen.get_gen().random((k, minmax.shape[1]))
    return numpy.tile(minmax[0, :], (k, 1)) + d*numpy.tile(minmax[1, :]-minmax[0, :], (k, 1))

# Compute distances between two sets of instances
def L2_distance(A, B):
    return numpy.vstack([numpy.sqrt(numpy.sum((A - numpy.tile(B[i,:], (A.shape[0], 1)))**2, axis=1)) for i in range(B.shape[0])]).T

def L1_distance(A, B):
    return numpy.vstack([numpy.sum(numpy.abs(A - numpy.tile(B[i,:], (A.shape[0], 1))), axis=1) for i in range(B.shape[0])]).T

# Calculate contingency matrix
def contingency_matrix(actual, predicted, weights=None):
    if weights is None:
        weights = numpy.ones(actual.shape[0], dtype=int)
    ac_int = actual.astype(int)
    prd_int = predicted.astype(int)
    nb_ac = numpy.maximum(2, numpy.max(ac_int)+1) + 1*numpy.any(ac_int == -1)
    nb_prd = numpy.maximum(2, numpy.max(prd_int)+1) + 1*numpy.any(prd_int == -1)
    counts = numpy.zeros((nb_prd, nb_ac, 2), dtype=type(weights[0]))
    for p,a,w in zip(prd_int, ac_int, weights):
        counts[p, a, 0] += 1
        counts[p, a, 1] += w
    return counts

# Calculate evaluation metrics from confusion matrix
def TPR_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0. # TRUE POSITIVE
    return (confusion_matrix[1,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def TNR_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0: return 0. # TRUE NEGATIVE
    return (confusion_matrix[0,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FPR_CM(confusion_matrix):
    if confusion_matrix[1,0] == 0: return 0. # FALSE POSITIVE
    return (confusion_matrix[1,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FNR_CM(confusion_matrix):
    if confusion_matrix[0,1] == 0: return 0. # FALSE NEGATIVE
    return (confusion_matrix[0,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def recall_CM(confusion_matrix):
    if confusion_matrix[1,1]==0: return 0.
    return confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
def precision_CM(confusion_matrix):
    if confusion_matrix[1,1]==0: return 0.
    return confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
def accuracy_CM(confusion_matrix):
    if confusion_matrix[1,1]+confusion_matrix[0,0]==0: return 0.
    return (confusion_matrix[1,1]+confusion_matrix[0,0])/(confusion_matrix[1,1]+confusion_matrix[1,0]+confusion_matrix[0,1]+confusion_matrix[0,0])
metrics_cm = {"TPR": TPR_CM, "TNR": TNR_CM, "FPR": FPR_CM, "FNR": FNR_CM,
              "recall": recall_CM, "precision": precision_CM, "accuracy": accuracy_CM}

# Computes the confusion matrix and evaluation measures, given two lists of labels, representing the ground-truth and the predictions
# >> evals, cm = get_CM_evals(numpy.array([1,1,1,1,1,0,0,0,0,0]), numpy.array([1,1,1,1,0,0,0,0,1,1]))
def get_CM_evals(actual, predicted, weights=None, vks=None):
    if vks is None:
        vks = metrics_cm.keys()
    cm = contingency_matrix(actual, predicted, weights)
    if weights is None:
        cm = cm[:, :, 0]
    else:
        cm = cm[:, :, 1]
    evals = {}
    for vk in vks:
        if vk in metrics_cm:
            evals[vk] = metrics_cm[vk](cm)
    return evals, cm


########################################################
#### SUPPORT VECTOR MACHINES (SVM)
########################################################
import cvxopt.solvers
# cvxopt.solvers.options['show_progress'] = False
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):

    kfunctions = {}
    kfeatures = {}

    def __init__(self, ktype='linear', kparams={}):
        self.ktype = 'linear'
        if ktype in self.kfunctions:
            self.ktype = ktype
        else:
            raise Warning("Kernel %s not implemented!" % self.ktype)
        self.kparams = kparams

    def distance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        return self.kfunctions[self.ktype](X, Y, **self.kparams)


def linear(X, Y): ### dot product in the original space, for the linear SVM
    return numpy.dot(X, Y.T)
Kernel.kfunctions['linear'] = linear

### kernel distance matrix (kdm) functions
#### K(x, x') = \sum_i (x^T x' + c_i) * d_i
def kdm_A(X, Y, degrees=None, offs=None):
    if degrees is None:
        return linear(X, Y)
    if offs is None or len(offs) != len(degrees):
    	offs = [0 for d in degrees]
    return numpy.sum(numpy.dstack([(numpy.dot(X, Y.T)+offs[i])**d for i,d in enumerate(degrees)]), axis=2)   
 
#### K(x, x') = \exp((||x-x'||**2)/(2*\sigma**2))
def kdm_B(X, Y, sigma):
    return numpy.vstack([numpy.exp(-numpy.sum((X-numpy.outer(numpy.ones(X.shape[0]), Y[yi,:]))** 2, axis=1) / (2. * sigma ** 2)).T for yi in range(Y.shape[0])]).T

### ... ### set the correct function for the following kernels
Kernel.kfunctions['RBF'] = kdm_B ### Radial Basis Function
Kernel.kfunctions['polynomial'] = kdm_A ### polynomial kernel

def compute_multipliers(X, y, c, kernel):
    n_samples, n_features = X.shape

    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(numpy.outer(y, y) * K)
    q = cvxopt.matrix(-1 * numpy.ones(n_samples))
    if c==0: ### hard margin: c == 0
        G = cvxopt.matrix(-1 * numpy.eye(n_samples))
        h = cvxopt.matrix(numpy.zeros(n_samples))
    else: ### soft margin: c > 0
        G = cvxopt.matrix(numpy.vstack((-1 * numpy.eye(n_samples), numpy.eye(n_samples))))
        h = cvxopt.matrix(numpy.hstack((numpy.zeros(n_samples), c * numpy.ones(n_samples))))
    A = cvxopt.matrix(numpy.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return numpy.ravel(solution['x'])

def svm_predict_vs(data, model):
    xx = model["kernel"].distance_matrix(model["support_vectors"], data)
    yy = model["lmbds"] * model["support_vector_labels"]
    return model["bias"] + numpy.dot(xx.T, yy)

def prepare_svm_model(X, y, c, ktype="linear", kparams={}):
    ### WARNING: SVM expect labels in {-1, 1} !
    ### the following line converts labels from {0, 1} to {-1, 1}
    y = 2.*y-1
    kernel = Kernel(ktype, kparams)

    lagrange_multipliers = compute_multipliers(X, y, c, kernel)
    support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

    model = {"kernel": kernel, "bias": 0.0,
             "lmbds": lagrange_multipliers[support_vector_indices],
             "support_vectors": X[support_vector_indices],
             "support_vector_labels": y[support_vector_indices]}
    pvs = svm_predict_vs(model["support_vectors"], model)
    ### ... ### bias = -(max prediction for positive support vector + min prediction for negative support vector)/2
    model["bias"] = -(max(pvs[model['support_vector_labels']>0]) + min(pvs[model['support_vector_labels']<0])) / 2

    return model, support_vector_indices


def visu_plot_svm(train_set, test_set, model, svi=None):
    minmax = dataset_minmax(numpy.vstack([train_set, test_set]))
    i, j = (0,1)
    gs = []
    lims = []
    for gi in range(train_set.shape[1]):
        step_size = float(minmax[1, gi]-minmax[0, gi])/100
        gs.append(numpy.arange(minmax[0, gi]-step_size, minmax[1, gi]+1.5*step_size, step_size))
        lims.append([minmax[0, gi]-2*step_size, minmax[1, gi]+2*step_size])
    axe = plt.subplot()

    bckgc = (0, 0, 0, 0)
    color = "#888888"
    color_lgt = "#DDDDDD"
    cmap="coolwarm"
    ws = numpy.dot(model["lmbds"]*model["support_vector_labels"], model["support_vectors"])
    print("weigths = ", ws)
    coeffs = numpy.hstack([ws, [model["bias"]]])
    sv_points = []

    vmin, vmax = (minmax[0,-1], minmax[1,-1])
    axe.scatter(train_set[:, j], train_set[:,i], c=train_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=50, marker=".", edgecolors='face', linewidths=2, gid="data_points_lbl")
    axe.scatter(test_set[:, j], test_set[:,i], c=test_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=55, marker="*", edgecolors='face', linewidths=2, zorder=2, gid="data_points_ulbl")


    if svi is not None:
        sv_points = numpy.where(svi)[0]

    xs = numpy.array([gs[j][0],gs[j][-1]])
    tmp = -(coeffs[i]*numpy.array([gs[i][0],gs[i][-1]])+coeffs[-1])/coeffs[j]
    xtr_x = numpy.array([numpy.maximum(tmp[0], xs[0]), numpy.minimum(tmp[-1], xs[-1])])

    x_pos = xtr_x[0]+.1*(xtr_x[1]-xtr_x[0])
    x_str = xtr_x[0]+.66*(xtr_x[1]-xtr_x[0])

    ys = -(coeffs[j]*xs+coeffs[-1])/coeffs[i]
    axe.plot(xs, ys, "-", color=color, linewidth=0.5, zorder=5, gid="details_svm_boundary")

    closest = (None, 0.)
    p0 = numpy.array([0, -coeffs[-1]/coeffs[i]])
    ff = numpy.array([1, -coeffs[j]/coeffs[i]])
    V = numpy.outer(ff, ff)/numpy.dot(ff, ff)
    offs = numpy.dot((numpy.eye(V.shape[0]) - V), p0)

    mrgs = [1.]
    for tii, ti in enumerate(sv_points):
        proj = numpy.dot(V, train_set[ti,[j, i]]) + offs
        axe.plot([train_set[ti,j], proj[0]], [train_set[ti,i], proj[1]], color=color_lgt, linewidth=0.25, zorder=0, gid=("details_svm_sv%d" % tii))

    #### plot margin
    for mrg in mrgs:
        yos = -(coeffs[j]*xs+coeffs[-1]-mrg)/coeffs[i]
        axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginA")
        yos = -(coeffs[j]*xs+coeffs[-1]+mrg)/coeffs[i]
        axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginB")

        mrgpx = x_pos # xs[0]+pos*(xs[-1]-xs[0])
        mrgpy = -(coeffs[j]*mrgpx+coeffs[-1]-mrg)/coeffs[i]
        mrgp = numpy.array([mrgpx, mrgpy])
        proj = numpy.dot(V, mrgp) + offs
        mrgv = numpy.sqrt(numpy.sum((mrgp-proj)**2))
        print("m=%.3f" % mrgv)
        axe.arrow(proj[0], proj[1], (proj[0]-mrgpx), (proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthA")
        axe.arrow(proj[0], proj[1], -(proj[0]-mrgpx), -(proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthB")


        axe.annotate("m=%.3f" % mrgv, (proj[0], proj[1]), (0, 15), textcoords='offset points', color=color, backgroundcolor=bckgc, zorder=10, gid="details_svm_mwidth-ann")
    axe.set_xlim(lims[j][0], lims[j][-1])
    axe.set_ylim(lims[i][0], lims[i][-1])
    plt.show()

def run_holdout_experiment_series(data_params, algo_params_series, ratio_train = .8, graph = 1):

    dataset, head, classes = load_csv(**data_params)

    ids = RandomNumGen.get_gen().permutation(dataset.shape[0])
    split_pos = int(len(ids)*ratio_train)
    train_ids, test_ids = ids[:split_pos], ids[split_pos:]
    train_set = dataset[train_ids]
    test_set = dataset[test_ids]

    for algo_params in algo_params_series:
        model, svi = prepare_svm_model(train_set[:,:-1], train_set[:,-1], **algo_params)
        t = svm_predict_vs(test_set[:,:-1], model)
        predicted = 1*(t>0)
        evals, cm = get_CM_evals(test_set[:, -1], predicted)
        ### visu_plot_svm (2D datasets only)
        print(evals)
        print(cm)
        print("bias = ", model["bias"])
        if graph == 1:
            visu_plot_svm(train_set, test_set, model, svi)
    