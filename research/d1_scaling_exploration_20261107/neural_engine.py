"""CPU float64 batched no-bias networks and frozen-initial-gradient twins.

This module creates no task, opens no file, and starts no experiment. Scalar
function-space clocks depend on the chosen initial gradient and parameterization;
their names do not imply coordinate invariance.
"""
from __future__ import annotations

import numpy as np
import torch


METHODS = ("SGD", "Momentum", "Adam", "AdamW", "RMSprop", "Adagrad")


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float64, device="cpu")


def _activation_pattern(activation, layers):
    pattern=[activation]*(layers-1) if isinstance(activation,str) else list(activation)
    if len(pattern)!=layers-1 or any(a not in ('identity','relu') for a in pattern):
        raise ValueError('activation must specify identity/relu for each hidden layer')
    return pattern


def _network_inputs(weights, X, activation):
    if not weights:
        raise ValueError("at least one weight matrix is required")
    _activation_pattern(activation,len(weights))
    weights = [_tensor(w) for w in weights]
    X = _tensor(X)
    if X.ndim != 3:
        raise ValueError("X must have shape (N,n,d)")
    ncases, nexamples, dim = X.shape
    if ncases < 1 or nexamples < 1 or dim < 1:
        raise ValueError("X dimensions must be nonempty")
    for w in weights:
        if w.ndim != 3 or w.shape[0] != ncases or w.shape[2] != dim or w.shape[1] < 1:
            raise ValueError("weights must be compatible (N,out,in) matrices")
        dim = w.shape[1]
    if dim != 1:
        raise ValueError("the final layer must have scalar output")
    return weights, X


def forward_batch(weights, X, activation="identity"):
    """Return (N,n) predictions; this differentiable function has no side effects."""
    h = X
    pattern=_activation_pattern(activation,len(weights))
    for index, w in enumerate(weights):
        h = torch.bmm(h, w.transpose(1, 2))
        if index < len(weights)-1 and pattern[index] == "relu":
            h = torch.relu(h)
    return h.squeeze(-1)


@torch.no_grad()
def make_initial_cache(weights0, X, activation="identity"):
    """Cache fixed weights, inputs, initial preactivations, and hidden activations."""
    weights, X = _network_inputs(weights0, X, activation)
    weights = [w.detach().clone() for w in weights]
    pattern=_activation_pattern(activation,len(weights))
    h = X.detach().clone()
    activations = [h]
    preactivations = []
    for index, w in enumerate(weights):
        z = torch.bmm(h, w.transpose(1, 2))
        preactivations.append(z)
        h = torch.relu(z) if index < len(weights)-1 and pattern[index] == "relu" else z
        activations.append(h)
    return {"weights": weights, "activations": activations,
            "preactivations": preactivations, "activation_pattern": pattern,
            "prediction": h.squeeze(-1)}


@torch.no_grad()
def jvp_from_initial(cache, delta_weights):
    """Exact J0*delta without materializing J0; ReLU derivative at zero is zero."""
    if len(delta_weights) != len(cache["weights"]):
        raise ValueError("delta_weights must match the cached layer count")
    tangent = torch.zeros_like(cache["activations"][0])
    for index, (w, dw) in enumerate(zip(cache["weights"], delta_weights)):
        dw = _tensor(dw)
        if dw.shape != w.shape:
            raise ValueError("delta weight shapes must match cached weights")
        tangent = (torch.bmm(tangent, w.transpose(1, 2))
                   + torch.bmm(cache["activations"][index], dw.transpose(1, 2)))
        if index < len(delta_weights)-1 and cache["activation_pattern"][index] == "relu":
            tangent = tangent * (cache["preactivations"][index] > 0)
    return tangent.squeeze(-1)


@torch.no_grad()
def initial_jvp(weights0, X, delta_weights, activation="identity"):
    """Convenience wrapper when a reusable cache is not needed."""
    return jvp_from_initial(make_initial_cache(weights0, X, activation), delta_weights)


@torch.no_grad()
def initial_kernel(weights0, X, activation="identity"):
    """Return exact per-case J0 J0^T:(N,n,n) via layerwise Gram products.

    For a layer the parameter Jacobian row is s_l outer h_(l-1), so its
    Gram is (s_l s_l^T) elementwise multiplied by (h_(l-1) h_(l-1)^T).
    Neither this kernel nor clocks built using g0 are parameterization invariant.
    """
    cache = make_initial_cache(weights0, X, activation)
    ncases, nexamples = cache["prediction"].shape
    kernel = torch.zeros((ncases, nexamples, nexamples), dtype=torch.float64)
    sensitivity = torch.ones_like(cache["preactivations"][-1])
    for index in range(len(cache["weights"])-1, -1, -1):
        hprev = cache["activations"][index]
        gram_s = torch.bmm(sensitivity, sensitivity.transpose(1, 2))
        gram_h = torch.bmm(hprev, hprev.transpose(1, 2))
        kernel += gram_s * gram_h
        if index > 0:
            sensitivity = torch.bmm(sensitivity, cache["weights"][index])
            if cache['activation_pattern'][index-1] == "relu":
                sensitivity *= cache["preactivations"][index-1] > 0
    return kernel


def _vector(value, ncases, name):
    value = _tensor(value)
    if value.ndim == 0:
        value = value.expand(ncases).clone()
    if value.shape != (ncases,) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite scalar or a per-case vector")
    return value


def _norm(weights):
    return torch.sqrt(sum(torch.sum(w*w, dim=(1, 2)) for w in weights))


def _finite(weights):
    result = torch.ones(weights[0].shape[0], dtype=torch.bool)
    for w in weights:
        result &= torch.isfinite(w).all(dim=2).all(dim=1)
    return result


def _ratio(numerator, denominator):
    result = torch.full_like(numerator, torch.nan)
    good = denominator > 0
    result[good] = numerator[good] / denominator[good]
    return result


class _Optimizer:
    """One vectorized optimizer state, one independent parameter set per case."""
    def __init__(self, weights, eta, method, parameters):
        self.method, self.parameters = method, parameters
        self.lr = eta[:, None, None]
        self.p = {key: value[:, None, None] for key, value in parameters.items()}
        self.state = {}
        if method == "Momentum":
            self.state["velocity"] = [torch.zeros_like(w) for w in weights]
        elif method in ("Adam", "AdamW"):
            self.state["m"] = [torch.zeros_like(w) for w in weights]
            self.state["v"] = [torch.zeros_like(w) for w in weights]
        elif method == "RMSprop":
            self.state["v"] = [torch.zeros_like(w) for w in weights]
        elif method == "Adagrad":
            self.state["v"] = [torch.ones_like(w)*self.p["adagrad_init"] for w in weights]

    @torch.no_grad()
    def step(self, weights, gradients, t):
        new = []
        p = self.p
        for index, (value, grad) in enumerate(zip(weights, gradients)):
            if self.method != "AdamW":
                grad = grad + p["weight_decay"]*value
            if self.method == "SGD":
                result = value - self.lr*grad
            elif self.method == "Momentum":
                velocity = self.state["velocity"][index]
                velocity.mul_(p["mu"]).add_(grad)
                result = value - self.lr*velocity
            elif self.method in ("Adam", "AdamW"):
                m, v = self.state["m"][index], self.state["v"][index]
                m.mul_(p["beta1"]).add_((1-p["beta1"])*grad)
                v.mul_(p["beta2"]).add_((1-p["beta2"])*grad*grad)
                mhat, vhat = m/(1-p["beta1"]**t), v/(1-p["beta2"]**t)
                if self.method == "AdamW":
                    value = value*(1-self.lr*p["weight_decay"])
                result = value-self.lr*mhat/(torch.sqrt(vhat)+p["eps"])
            else:
                v = self.state["v"][index]
                if self.method == "RMSprop":
                    v.mul_(p["alpha"]).add_((1-p["alpha"])*grad*grad)
                else:
                    v.add_(grad*grad)
                result = value-self.lr*grad/(torch.sqrt(v)+p["eps"])
            new.append(result.detach())
        return new

    def finite(self):
        if not self.state:
            return torch.ones(len(self.lr), dtype=torch.bool)
        result = torch.ones(len(self.lr), dtype=torch.bool)
        for arrays in self.state.values():
            result &= _finite(arrays)
        return result


def simulate_batch(weights0, X, y, Xval, yval, eta, method,
                   activation="identity", steps=256, optimizerparams=None):
    """Simulate real and constant-g0 network updates; all layers have no bias.

    Scalar trajectories are NumPy float64 arrays (N,steps+1), masks are bool.
    Initial/final weight and gradient lists are detached CPU float64 tensors.
    Initial loss uses .5*mean over each case's examples, with no case averaging.
    Optional Xval/yval must both be None or both supplied. A real case stops at
    its first nonfinite state/train/validation loss or train loss >1e10*L0.
    Frozen cases stop only on nonfinite diagnostics/state. Invalid samples and
    all subsequent samples are NaN, while other cases continue independently.
    Net displacement is reported, not cumulative path length. Zero parameter or
    residual norms yield NaN in the corresponding normalized clock, not failure.
    """
    initial, X = _network_inputs(weights0, X, activation)
    initial = [w.detach().clone() for w in initial]
    ncases, nexamples, _ = X.shape
    y = _tensor(y)
    if y.shape != (ncases, nexamples):
        raise ValueError("y must have shape (N,n)")
    if (Xval is None) != (yval is None):
        raise ValueError("Xval/yval must both be supplied or both omitted")
    if Xval is not None:
        _, Xval = _network_inputs(initial, Xval, activation)
        yval = _tensor(yval)
        if yval.shape != Xval.shape[:2]:
            raise ValueError("yval must match Xval's case/example dimensions")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    eta = _vector(eta, ncases, "eta")
    if bool((eta < 0).any()):
        raise ValueError("eta must be nonnegative")
    defaults = {"mu":.9,"beta1":.9,"beta2":.999,"alpha":.99,"eps":1e-8,
                "weight_decay":.01 if method == "AdamW" else 0.,"adagrad_init":0.}
    supplied = {} if optimizerparams is None else dict(optimizerparams)
    if set(supplied)-set(defaults):
        raise ValueError(f"unsupported optimizer parameters: {set(supplied)-set(defaults)}")
    params = {key:_vector(supplied.get(key,value),ncases,key) for key,value in defaults.items()}
    for key in ("mu","beta1","beta2","alpha"):
        if bool(((params[key] < 0) | (params[key] >= 1)).any()):
            raise ValueError(f"{key} must lie in [0,1)")
    for key in ("eps","weight_decay","adagrad_init"):
        if bool((params[key] < 0).any()):
            raise ValueError(f"{key} must be nonnegative")

    real = [w.clone().requires_grad_(True) for w in initial]
    frozen = [w.clone() for w in initial]
    cache = make_initial_cache(initial, X, activation)
    residual0 = cache["prediction"]-y
    residual_norm = torch.linalg.vector_norm(residual0, dim=1)
    param_norm = _norm(initial)
    prediction = forward_batch(real, X, activation)
    losses = .5*torch.mean((prediction-y)**2, dim=1)
    gradients = torch.autograd.grad(losses.sum(), real)
    initial_grads = [g.detach().clone() for g in gradients]
    gradient_norm = _norm(initial_grads)
    initial_loss = losses.detach().clone()
    real_optimizer = _Optimizer(initial, eta, method, params)
    frozen_optimizer = _Optimizer(initial, eta, method, params)
    names = ["loss","validation_loss","u_const","relative_d","function_norm_V",
             "function_projection_P","u_actual","exact_const_loss"]
    trajectories = {name:np.full((ncases,steps+1),np.nan,dtype=np.float64) for name in names}
    valid_real = np.zeros((ncases,steps+1),dtype=bool)
    valid_const = np.zeros((ncases,steps+1),dtype=bool)
    bad_real = np.full(ncases,-1,dtype=np.int64)
    bad_const = np.full(ncases,-1,dtype=np.int64)
    active = _finite(initial)&_finite(initial_grads)&torch.isfinite(initial_loss)&torch.isfinite(gradient_norm)
    const_active = active.clone()
    initial_validation_prediction = None

    def validation(weights):
        if Xval is None:
            return torch.full((ncases,),torch.nan,dtype=torch.float64)
        return .5*torch.mean((forward_batch(weights,Xval,activation)-yval)**2,dim=1)

    with torch.no_grad():
        initial_validation_prediction = None if Xval is None else forward_batch(initial,Xval,activation).clone()
        vl = validation(initial)
        if Xval is not None:
            active &= torch.isfinite(vl)
        bad_real[~active.numpy()] = 0
        bad_const[~const_active.numpy()] = 0
        valid_real[:,0] = active.numpy()
        valid_const[:,0] = const_active.numpy()
        for name,value,mask in [("loss",initial_loss,active),("validation_loss",vl,active),
                                ("exact_const_loss",initial_loss,const_active)]:
            trajectories[name][mask.numpy(),0] = value[mask].numpy()
        for name,mask in [("u_actual",active),("u_const",const_active)]:
            trajectories[name][mask.numpy(),0] = 0.
        trajectories["relative_d"][(const_active&(param_norm>0)).numpy(),0] = 0.
        for name in ("function_norm_V","function_projection_P"):
            trajectories[name][(const_active&(residual_norm>0)).numpy(),0] = 0.

    for t in range(1,steps+1):
        # The first real gradients have already been calculated; later ones are
        # derived from only the current real model. The twin never recomputes g0.
        if t > 1:
            gradients = torch.autograd.grad(losses.sum(), real)
        next_real = real_optimizer.step(real, gradients, t)
        next_frozen = frozen_optimizer.step(frozen, initial_grads, t)
        # Reset already failed rows internally, without mutating tensors retained
        # by the next backward pass; their saved trajectories remain NaN.
        next_real=[torch.where(active[:,None,None],w,w0) for w,w0 in zip(next_real,initial)]
        next_frozen=[torch.where(const_active[:,None,None],w,w0) for w,w0 in zip(next_frozen,initial)]
        real = [w.detach().requires_grad_(True) for w in next_real]
        frozen = next_frozen
        prediction = forward_batch(real, X, activation)
        losses = .5*torch.mean((prediction-y)**2,dim=1)
        with torch.no_grad():
            vl = validation(real)
            delta_real = [w-w0 for w,w0 in zip(real,initial)]
            delta_const = [w-w0 for w,w0 in zip(frozen,initial)]
            ur, uc = _norm(delta_real), _norm(delta_const)
            tangent = jvp_from_initial(cache,delta_const)
            function_norm = _ratio(torch.linalg.vector_norm(tangent,dim=1),residual_norm)
            function_projection = _ratio(-torch.sum(residual0*tangent,dim=1),residual_norm**2)
            exact_const = .5*torch.mean((forward_batch(frozen,X,activation)-y)**2,dim=1)
            good = (active&_finite(real)&real_optimizer.finite()&torch.isfinite(losses)
                    &torch.isfinite(ur)&(losses<=1e10*initial_loss))
            if Xval is not None:
                good &= torch.isfinite(vl)
            good_const = (const_active&_finite(frozen)&frozen_optimizer.finite()&torch.isfinite(uc)
                          &torch.isfinite(exact_const)&torch.isfinite(tangent).all(dim=1))
            bad_real[(active&~good).numpy()] = t
            bad_const[(const_active&~good_const).numpy()] = t
            valid_real[:,t],valid_const[:,t] = good.numpy(),good_const.numpy()
            values = {"loss":losses.detach(),"validation_loss":vl,"u_actual":ur,
                      "u_const":uc,"relative_d":_ratio(uc,param_norm),
                      "function_norm_V":function_norm,"function_projection_P":function_projection,
                      "exact_const_loss":exact_const}
            for name,value in values.items():
                mask = good if name in ("loss","validation_loss","u_actual") else good_const
                trajectories[name][mask.numpy(),t] = value[mask].numpy()
            active,const_active = good,good_const

    result = {**trajectories,"loss_const":trajectories["exact_const_loss"],
              "valid_actual":valid_real,"valid_const":valid_const,
              "diverged_actual":bad_real>=0,"diverged_const":bad_const>=0,
              "first_divergence_step_actual":bad_real,"first_divergence_step_const":bad_const,
              "initial_gradient_norm":gradient_norm.detach().numpy().copy(),
              "initial_param_norm":param_norm.numpy().copy(),
              "initial_residual_norm":residual_norm.numpy().copy(),
              "initial_loss":initial_loss.numpy().copy(),
              "initial_prediction":cache["prediction"].clone(),
              "initial_validation_prediction":initial_validation_prediction,
              "initial_residual":residual0.clone(),
              "initial_grads":[g.clone() for g in initial_grads],
              "final_weights_actual":[torch.where(active[:,None,None],w.detach(),torch.nan).clone() for w in real],
              "final_weights_const":[torch.where(const_active[:,None,None],w,torch.nan).clone() for w in frozen]}
    return result
