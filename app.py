from optimization.bayesian_optimizer import BayesianOptimizer
from risk.kelly import KellySizer
from macro.factor_model import MacroFactorModel
from execution.rl_execution import RLExecutionAgent
st.sidebar.header("Advanced Controls")

if st.sidebar.button("Run Bayesian Optimization"):
    best_params, best_score = optimizer.optimize(30)
    st.write("Best Params:", best_params)
    st.write("Best Sharpe:", best_score)
