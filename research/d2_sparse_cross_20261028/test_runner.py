"""Synthetic checks for step derivatives and retained-pair risk arithmetic."""
import importlib.util
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('sparse_d2_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(s);s.loader.exec_module(runner)


def test_step_derivative_has_correct_eta():
    data={'phi':np.array([[.4,.1],[.1,.3],[-.2,.2]]),'val_phi':np.eye(2)}
    model=runner.legacy.spectral_model(data)
    t=np.array([.2,3.,20.]);eps=1e-4
    _,df=runner.f_and_dt(model,t)
    plus,_=runner.legacy.response(model,t+eps)
    minus,_=runner.legacy.response(model,t-eps)
    np.testing.assert_allclose(df,(plus-minus)/(2*eps),atol=1e-10,rtol=1e-8)


def test_cross_example_and_zero_mask():
    # Contributions 2 and -1, correct target 1: full error 0; deleting cross gives 4.
    f=np.array([[2.,-1.]])
    terms={'linear':np.ones(2),'val_y2':1.}
    h=np.ones((2,2));pairs=runner.pair_indices(np.ones(2,dtype=bool))
    empty=runner.masked_h(h,pairs,np.array([],dtype=int))
    full=runner.masked_h(h,pairs,np.arange(len(pairs)))
    np.testing.assert_allclose(runner.signal_curve(f,full,terms),[0])
    np.testing.assert_allclose(runner.signal_curve(f,empty,terms),[4])
    np.testing.assert_array_equal(empty,np.diag(np.diag(h)))
