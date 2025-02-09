

print("starting imports")
import numpy as np
print("skopt")
from skopt import Optimizer
from skopt.acquisition import _gaussian_acquisition
from skopt.space import Real, Integer, Categorical
from skopt.plots import _evenly_sample
print("done with imports")


class GaussianProcessOptimizer:
    def __init__(self, search_space, previous_results):
        """
        Initializes the optimizer with a given search space and previous results.

        search_space: List of skopt.space dimensions (defines the range of hyperparameters)
        previous_results: List of dicts, each containing hyperparameter values and a 'score' (lower is better)
        min_dist: Minimum Euclidean distance between suggested points to ensure diversity
        """
        self.search_space = search_space
        self.previous_results = previous_results
        self._prepare_data()

        self.optimizer = Optimizer(dimensions=self.search_space, base_estimator="GP", random_state=1, n_initial_points=10)

        for x, y in zip(self.X, self.y):
            self.optimizer.tell(x, y)

    def _prepare_data(self):
        self.X = [list(d.values())[:-1] for d in self.previous_results]  # Extract hyperparameter values
        self.y = [d['score'] for d in self.previous_results]  # Extract corresponding scores

    def suggest_next_points(self, n_points=16):
        new_points = self.optimizer.ask(n_points, strategy="cl_mean")
        return [dict(zip([dim.name for dim in self.search_space], vals)) for vals in new_points]

# from scikit-optimize
def plot_gaussian_process(
    res,
    start_point,
    dim=0,
    ax=None,
    n_calls=-1,
    objective=None,
    n_points=1000,
    noise_level=0,
    show_legend=True,
    show_title=True,
    show_acq_func=False,
    show_next_point=False,
    show_observations=True,
    show_mu=True,
):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    #n_dims = res.space.n_dims
    #assert n_dims == 1, "Space dimension must be 1"
    assert res.space.n_dims == len(start_point), "Dimension mismatch"
    dimension = res.space.dimensions[dim]
    x, x_model = _evenly_sample(dimension, n_points)
    x = x.reshape(-1, 1)
    x_model = x_model.reshape(-1, 1)

    x_full_data = [start_point.copy() for _ in range(n_points)]
    for i in range(n_points):
        x_full_data[i][dim] = x[i][0]
    x = np.array([x[dim] for x in x_full_data]) # , dtype=np.float64
    x_model = res.space.transform(x_full_data)

    if res.specs is not None and "args" in res.specs:
        n_random = res.specs["args"].get('n_random_starts', None)
        acq_func = res.specs["args"].get("acq_func", "EI")
        acq_func_kwargs = res.specs["args"].get("acq_func_kwargs", {})

    if acq_func_kwargs is None:
        acq_func_kwargs = {}
    if acq_func is None or acq_func == "gp_hedge":
        acq_func = "EI"
    if n_random is None:
        n_random = len(res.x_iters) - len(res.models)

    if objective is not None:
        fx = np.array([objective(x_i) for x_i in x_full_data])
    if n_calls < 0:
        model = res.models[-1]
        curr_x_iters = res.x_iters
        curr_func_vals = res.func_vals
    else:
        model = res.models[n_calls]

        curr_x_iters = res.x_iters[: n_random + n_calls]
        curr_func_vals = res.func_vals[: n_random + n_calls]
    curr_x_iters = [x[dim] for x in curr_x_iters]
    # print(list(zip(curr_x_iters, curr_func_vals.tolist())))


    if objective is not None:
        ax.plot(x, fx, "r--", label="True (unknown)")
        ax.fill(
            np.concatenate([x, x[::-1]]),
            np.concatenate(
                (
                    [fx_i - 1.9600 * noise_level for fx_i in fx],
                    [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]],
                )
            ),
            alpha=0.2,
            fc="r",
            ec="None",
        )


    if show_mu:
        per_second = acq_func.endswith("ps")
        if per_second:
            y_pred, sigma = model.estimators_[0].predict(x_model, return_std=True)
        else:
            y_pred, sigma = model.predict(x_model, return_std=True)
        ax.plot(x, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        ax.fill(
            np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=0.2,
            fc="g",
            ec="None",
        )


    if show_observations:
        ax.scatter(curr_x_iters, curr_func_vals, color="red", s=20, label="Observations (scatter)") 
        #ax.plot  (curr_x_iters, curr_func_vals, "r.", markersize=8, label="Observations")
    if (show_mu or show_observations or objective is not None) and show_acq_func:
        ax_ei = ax.twinx()
        ax_ei.set_ylabel(str(acq_func) + "(x)")
        plot_both = True
    else:
        ax_ei = ax
        plot_both = False
    if show_acq_func:
        acq = _gaussian_acquisition(
            x_model,
            model,
            y_opt=np.min(curr_func_vals),
            acq_func=acq_func,
            acq_func_kwargs=acq_func_kwargs,
        )
        next_x = x[np.argmin(acq)]
        next_acq = acq[np.argmin(acq)]
        acq = -acq
        next_acq = -next_acq
        ax_ei.plot(x, acq, "b", label=str(acq_func) + "(x)")
        if not plot_both:
            ax_ei.fill_between(x.ravel(), 0, acq.ravel(), alpha=0.3, color='blue')

        if show_next_point and next_x is not None:
            ax_ei.plot(next_x, next_acq, "bo", markersize=6, label="Next query point")

    if show_title:
        # make res.x into list of dim.name: val
        s = ", ".join([f"{dim.name}: {val}" for dim, val in zip(res.space.dimensions, res.x)])
        ax.set_title(fr"x* = {s}, \n f(x*) = {res.fun:.4f}")

    ax.grid()
    ax.set_xlabel("x" if dimension.name is None else dimension.name)
    ax.set_ylabel("f(x)")
    if dimension.prior == "log-uniform":
        ax.set_xscale("log")
    if show_legend:
        if plot_both:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_ei.get_legend_handles_labels()
            ax_ei.legend(
                lines + lines2,
                labels + labels2,
                loc="best",
                prop={'size': 6},
                numpoints=1,
            )
        else:
            ax.legend(loc="best", prop={'size': 6}, numpoints=1)

    return ax


if __name__ == "__main__":
    print("matplotlib")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    search_space = [
        Real(0.0001, 0.1, name='learning_rate', prior='log-uniform'),
        Integer(16, 128, name='batch_size'),  
        Categorical([True, False], name='use_batch_norm')  #
    ]

    previous_results = [
        {'learning_rate': 0.001, 'batch_size': 32, 'use_batch_norm': True, 'score': 0.8},
        {'learning_rate': 0.01, 'batch_size': 64, 'use_batch_norm': False, 'score': 0.75},
        {'learning_rate': 0.005, 'batch_size': 128, 'use_batch_norm': True, 'score': 0.76},
        {'learning_rate': 0.001, 'batch_size': 32, 'use_batch_norm': True, 'score': 0.81},
        {'learning_rate': 0.01, 'batch_size': 55, 'use_batch_norm': False, 'score': 0.75},
        {'learning_rate': 0.005, 'batch_size': 56, 'use_batch_norm': True, 'score': 0.78},
        {'learning_rate': 0.001, 'batch_size': 67, 'use_batch_norm': True, 'score': 0.8},
        {'learning_rate': 0.01, 'batch_size': 98, 'use_batch_norm': False, 'score': 0.75},
        {'learning_rate': 0.005, 'batch_size': 22, 'use_batch_norm': True, 'score': 0.78},
        {'learning_rate': 0.1, 'batch_size': 22, 'use_batch_norm': True, 'score': 0.88},
        {'learning_rate': 0.001, 'batch_size': 30, 'use_batch_norm': True, 'score': 0.8},
        {'learning_rate': 0.01, 'batch_size': 32, 'use_batch_norm': False, 'score': 0.75},
        {'learning_rate': 0.005, 'batch_size': 63, 'use_batch_norm': True, 'score': 0.78}
    ]

    optimizer = GaussianProcessOptimizer(search_space, previous_results)

    plot_gaussian_process(optimizer.optimizer.get_result(), [0.01, 98, True], dim=2)

    import os
    fname = "automl"
    i = 1
    while os.path.exists(fname + ".png"):
        fname = "automl_" + str(i)
        i += 1
    print(fname)
    plt.savefig(fname + ".png")
