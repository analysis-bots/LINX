# LINX
This repository contains the source code and experiments used to evaluate LINX, a framework for auto-generating personalized exploration notebook using Natural Language Interface. 
The repository is free for use for academic purposes. Please contact the repository owners before usage.

## The problem: Auto-generating meaningful and relevant Exploratory Notebooks using a friendly interface
One of the most effective methods for facilitating the process of exploring a dataset is to examine existing data exploration notebooks prepared by other data analysts or scientists. These notebooks contain curated sessions of contextually-related query operations that all together demonstrate interesting hypotheses and conjectures on the data. Unfortunately, relevant such notebooks, that had been prepared on the same dataset, and in light of thesame analysis task – are often nonexistent or unavailable. LINX is a CDRL framework guided by an LLM component for autogenerating interesting, task-relevant notebooks given a user-provided dataset and task in natural language.  

## [Source Code](src)
The source code is located [here](src) (LINX/src) <br/>
Under this directory, there are two folders:
1. Exploration Plan Generator: contains all the source code of the LLM component which translates the user task in natural language to LDX specifications fed to the engine.
2. CDRL Engine: contains all the source code of the CDRL engine that generating the personalized exploration notebooks.
For installation guide, running instructions and further details please refer to the 
documentation under the source code directory in the link above.

## [Documentation](documentation)
1. [LINX Technical Report](documentation/LINX_Full_Paper.pdf) -  [here](documentation/LINX_Full_Paper.pdf).
2. [LDX Technical User Guide](documentation/LDX_User_Guide.pdf)
A simple user guide for writing LDX specifications is located [here](documentation/LDX_User_Guide.pdf). <br/>

## [Experiment Datasets](datasets)
The datasets used in our empirical evaluation are located [here](datasets). <br/>
LINX is tested on 3 different datasets:
1. Netflix Movies and TV-Shows: list of Netflix titles, each title is described using 11 features such as the country of production, duration/num. of seasons, etc.
2. Flight-delays: Each record describes a domestic US flight, using 12 attributes such as origin/destination airport, flight duration, issuing airline, departure delay times, delay reasons, etc.
3. Google Play Store Apps: A collection of mobile apps available on the Google Play Store. Each app is described using 11 features, such as name, category, price, num. of installs, reviews, etc.

## [NL2LDX Benchmark](nl2ldx_benchmark)
Our benchmark of NL-to-LDX translation tasks is located [here](<nl2ldx_benchmark/NL2LDX-benchmark.json>), the format of each tuple is as follows: [meta-task id, task, expected LDX, dataset]. <br/>
This folder includes also a dedicated notebook for evaluating models on this benchmark, located [here](<nl2ldx_benchmark/evaulation/evaluation_notebook.ipynb>), 
and as well the evaluation results of our models.

## [Additional Experiments](additional_experiments)
This folder contains information about:
1. [User Study](additional_experiments/user_study) - The exploration notebooks generated by either LINX and the baselines are located [here](additional_experiments/user_study). <br/>
In the given link you can find the exploratory sessions that were presented to each participant of the user study.
The directory structure is as: `<Dataset>/<Task>/<Baseline>.ipynb` (the identity of the baseline wasn't given to the participants).
For the ChatGPT-based notebooks, we also provide the prompt and raw output. 
2. [Convergence Test](additional_experiments/convergence) - The convergence and running times of our CDRL engine located [here](additional_experiments/convergence).


