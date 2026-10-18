import numpy as np
import pandas as pd
from scipy.optimize import minimize


def loadings(maturities, lambda1=0.0609, lambda2=None):
    tau = np.asarray(maturities, dtype=float)
    z = lambda1 * tau
    a = -np.expm1(-z) / z
    b = a - np.exp(-z)
    columns = [np.ones_like(tau), a, b]
    if lambda2 is not None:
        z2 = lambda2 * tau
        columns.append(-np.expm1(-z2) / z2 - np.exp(-z2))
    return np.column_stack(columns)


def extract_factors(yields, maturities, lambda1=0.0609, lambda2=None):
    basis = loadings(maturities, lambda1, lambda2)
    coefficients = np.linalg.lstsq(basis, yields[[str(m) for m in maturities]].T,
                                   rcond=None)[0].T
    return pd.DataFrame(coefficients, index=yields.index,
                        columns=[f"beta{i}" for i in range(basis.shape[1])])


def _swarm_loss(particles, target, tau):
    z1 = particles[:, 4, None] * tau[None, :]
    z2 = particles[:, 5, None] * tau[None, :]
    a1 = -np.expm1(-z1) / z1
    b1 = a1 - np.exp(-z1)
    b2 = -np.expm1(-z2) / z2 - np.exp(-z2)
    fitted = (particles[:, 0, None] + particles[:, 1, None] * a1
              + particles[:, 2, None] * b1 + particles[:, 3, None] * b2)
    return np.sum((fitted - target[None, :]) ** 2, axis=1)


def fit_nss_pso(target, maturities, settings, seed=42, previous=None, zlb=False):
    """Six-parameter local-ring PSO, then L-BFGS-B, then conditional OLS.

    Unreported PSO settings are explicit in config. Local search uses an ordered
    decay reparameterization so L-BFGS-B obeys lambda1 >= lambda2 > 0.
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(target, dtype=float)
    tau = np.asarray(maturities, dtype=float)
    count, iterations = settings["particles"], settings["iterations"]
    lower = np.array([-1., -1., -1., -1., 1e-9, 1e-9])
    upper = np.array([1., 1., 1., 1., .4, .4])
    span = upper - lower

    def project(p):
        p = np.clip(p, lower, upper)
        p[:, 4:] = -np.sort(-p[:, 4:], axis=1)
        if zlb:
            p[:, 0] = np.maximum(p[:, 0], 0)
            p[:, 1] = np.maximum(p[:, 1], -p[:, 0])
        return p

    positions = project(rng.uniform(lower, upper, size=(count, 6)))
    anchor = np.r_[np.linalg.lstsq(loadings(tau, .0609, .01), target, rcond=None)[0], .0609, .01]
    positions[0] = project(anchor[None, :])[0]
    if previous is not None:
        positions[1] = project(np.asarray(previous)[None, :])[0]
    velocity = rng.uniform(-.05, .05, positions.shape) * span
    personal = positions.copy()
    scores = _swarm_loss(personal, target, tau)
    best_score = float(scores.min())
    stall = 0
    initial_radius = max(1, int(np.ceil(.10 * count / 2)))
    indices = np.arange(count)
    for step in range(iterations):
        fraction = step / max(iterations - 1, 1)
        radius = min(count // 2, int(initial_radius + fraction * (count // 2 - initial_radius)))
        neighborhoods = (indices[:, None] + np.arange(-radius, radius + 1)[None, :]) % count
        best_neighbors = neighborhoods[indices, np.argmin(scores[neighborhoods], axis=1)]
        inertia = settings["inertia_start"] + fraction * (settings["inertia_end"] - settings["inertia_start"])
        velocity = (inertia * velocity
                    + settings["c1"] * rng.random(positions.shape) * (personal - positions)
                    + settings["c2"] * rng.random(positions.shape) * (personal[best_neighbors] - positions))
        vmax = settings["velocity_fraction"] * span
        velocity = np.clip(velocity, -vmax, vmax)
        positions = project(positions + velocity)
        current = _swarm_loss(positions, target, tau)
        improved = current < scores
        personal[improved], scores[improved] = positions[improved], current[improved]
        new_best = float(scores.min())
        stall = stall + 1 if best_score - new_best <= settings["stall_tolerance"] else 0
        best_score = new_best
        if stall >= settings["stall_iterations"]:
            break

    candidates = personal[np.argsort(scores)[:settings["local_particles"]]]

    def physical(q):
        q = np.asarray(q)
        betas = q[:4].copy()
        if zlb:
            # Endpoint representation: beta1 = short-end limit - beta0.
            betas[1] = q[1] - q[0]
        return np.r_[betas, q[4], 1e-9 + (q[4] - 1e-9) * q[5]]

    def objective_gradient(q):
        p = physical(q)
        basis = loadings(tau, p[4], p[5])
        residual = basis @ p[:4] - target
        grad = np.zeros(6)
        grad[:4] = 2 * basis.T @ residual
        gradients_lambda = []
        for lam, first in ((p[4], True), (p[5], False)):
            z = lam * tau
            da = tau * ((z + 1) * np.exp(-z) - 1) / z ** 2
            # Avoid cancellation very close to zero.
            small = np.abs(z) < 1e-4
            da[small] = tau[small] * (-.5 + z[small] / 3 - z[small] ** 2 / 8)
            db = da + tau * np.exp(-z)
            dy = p[1] * da + p[2] * db if first else p[3] * db
            gradients_lambda.append(2 * residual @ dy)
        grad[4] = gradients_lambda[0] + q[5] * gradients_lambda[1]
        grad[5] = (q[4] - 1e-9) * gradients_lambda[1]
        if zlb:
            grad[0] -= grad[1]
        # The default unscaled SSE preserves the paper's numerical tolerances.
        # Objective scaling is an explicit sensitivity because it changes the
        # effective absolute ftol/gtol, despite preserving the mathematical optimum.
        scale = float(settings.get("objective_scale", 1.0))
        return float(residual @ residual) * scale, grad * scale

    refined = []
    for p in candidates:
        q = p.copy()
        q[5] = (p[5] - 1e-9) / max(p[4] - 1e-9, 1e-12)
        if zlb:
            q[1] = np.clip(p[0] + p[1], 0, 1)
        bounds = [(-1, 1)] * 4 + [(1e-9, .4), (0, 1)]
        if zlb:
            bounds[:2] = [(0, 1), (0, 1)]
        result = minimize(objective_gradient, q, method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        refined.append(physical(result.x))
    all_candidates = np.vstack([candidates, refined])
    best = all_candidates[np.argmin(_swarm_loss(all_candidates, target, tau))]
    # The paper explicitly re-extracts factors by unrestricted OLS.
    beta_ols = np.linalg.lstsq(loadings(tau, best[4], best[5]), target, rcond=None)[0]
    return np.r_[beta_ols, best[4:]], best, step + 1


def build_curve_panels(yields, maturities, settings, output_dir, include_pso=True):
    from pathlib import Path
    output_dir = Path(output_dir)
    panels = {}
    for name, l1, l2 in (("DNS", .0609, None), ("DNSS", .0609, .01),
                         ("DNSS-fixedPSO", .035, .006)):
        factors = extract_factors(yields, maturities, l1, l2)
        panels[name] = {"factors": factors, "decays": np.tile([l1, l2 or np.nan], (len(yields), 1))}
        factors.to_csv(output_dir / f"factors_{name}.csv", index_label="month")
    if include_pso:
        for zlb in (False, True):
            name = "DNSS-PSO-ZLB" if zlb else "DNSS-PSO"
            cache = output_dir / f"factors_{name}.csv"
            if cache.exists():
                frame = pd.read_csv(cache, index_col=0)
                frame.index = pd.PeriodIndex(frame.index, freq="M")
                if not frame.index.equals(yields.index):
                    raise ValueError("Cached PSO dates do not match input data; use a fresh results directory")
            else:
                records, previous = [], None
                for i, (month, row) in enumerate(yields[[str(m) for m in maturities]].iterrows()):
                    fitted, previous, iterations = fit_nss_pso(row.to_numpy(), maturities, settings,
                                                               settings["seed"] + i, previous, zlb)
                    records.append([*fitted, iterations])
                    if (i + 1) % 50 == 0:
                        print(f"Curve calibration {name}: {i + 1}/{len(yields)}", flush=True)
                frame = pd.DataFrame(records, index=yields.index,
                                     columns=["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2", "iterations"])
                frame.to_csv(cache, index_label="month")
            panels[name] = {"factors": frame.iloc[:, :4], "decays": frame[["lambda1", "lambda2"]].to_numpy()}
    return panels

