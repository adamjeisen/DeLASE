# DeLASE: Delayed Linear Analysis for Stability Estimation
## A data-driven tool for robustly and rigorously quantifying changes in stability in complex partially-observed nonlinear dynamical systems.

This repository contains code for running the DeLASE algorithm (Delayed Linear Analysis for Stability Estimation). DeLASE was designed to quantify changes in population-level stability in complex systems. DeLASE harnesses the power of Koopman operators (namely, the HAVOK algorithm<sup>1</sup>) to efficiently generate linear representations of nonlinear partially-observed dynamics. DeLASE then reformulates the Koopman operator as a delay dynamical system, and utilizes tools (namely, the TRACE-DDE algorithm) from delay differential equations theory for estimating the stability of such equations<sup>2</sup>.

Please don't hestiate to reach out with any questions.

1. Brunton, S.L., Brunton, B.W., Proctor, J.L., Kaiser, E., and Kutz, J.N. (2017). Chaos as an intermittently forced linear system. Nat. Commun. 8, 19.
2. Breda, D., Maset, S., and Vermiglio, R. (2009). TRACE-DDE: a Tool for Robust Analysis and Characteristic Equations for Delay Differential Equations. In Topics in Time Delay Systems: Analysis, Algorithms and Control, J. J. Loiseau, W. Michiels, S.-I. Niculescu, and R. Sipahi, eds. (Springer Berlin Heidelberg), pp. 145–155.

## Install the repo using `pip`:

```
git clone https://github.com/adamjeisen/DeLASE
cd DeLASE/
python -m pip install --editable .
```

