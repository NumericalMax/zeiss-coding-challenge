# Zeiss Coding Challenge: Data Science Position

_Author: Maximilian Kapsecker_ <br>
_Challenge: Handling gaps in time series dataset from temperature readings for the development of anomaly detection algorithm_

## General Notes

I would like to share some general notes on my approach to Data Science tasks and the principles I follow to maximize the value derived from data, specifically to understand the design decision made for this coding challenge.

- **Understand the problem:** Achieving the best value from data is highly dependent on the end-user for whom the data science is conducted. Therefore, I typically begin by discussing the problem statement with a domain expert. This helps me understand the motivation, problem, and objectives in detail. For the underyling coding task, instead, I will mark the assumptions and simplifications made by myself which would usually be subjected to dicussing with a domain expert.

- **Separation of Concerns:** Jupyter Notebooks are an excellent tool for embedding data analysis within a narrative. However, including large amounts of (boilerplate) code in a notebook can detract from the storytelling purpose. To address this, I maintain separate Python code in files.

- **Simplicity over complexity**: Often, the best solutions are simple. While there are many advancements in data science that are worth considering, starting with a simple solution and then increasing complexity can be more effective. This approach helps in developing a good initial solution that can be iteratively improved.

- For the task in place, the notebook might seem to be overstructured (e.g., comments) and could be streamlined. However, it is intended to showcase my methodology and coding skills. By giving the project additional structure, I aim to provide a clear and comprehensive view of my approach.


## Structure

1. ```timeseries-analysis.ipynb```: contains the primary analysis of the data and is the main file for reviewing this work.
2. ```src``` folder: contains supporting code files. These files help keep the notebook clean and manageable by offloading complex code.
3. ```test``` folder: includes unittests for the implemented functions. **Caution**: not all functions have unittests due to time constraints.

## Install, Run and Test

To reproduce the results of the analysis pipeline, follow these steps:

1. Place your data into the 
1. Ensure you have a Python 3.10 environment set up.
Install the required dependencies:
```
pip install -r requirements.txt
```
2. Launch Jupyter Lab:
```
jupyter-lab
```
3. Open the ```timeseries-analysis.ipynb``` notebook in Jupyter Lab.
4. You can run the test by calling ```python -m unittest ./tests/test_helper.py```

## Open Tasks

I want to draw attention to some open TODOs that I couldn't manage to complete in an acceptable timeframe:

1. **Introduce a dedicated class for a timeseries**: This class should encapsulate functionality to modify the series as needed for the task, fostering sustainability by introducing a reusable component.
2. **Add additional test cases**: This will help achieve higher test coverage.
3. **Improve the handling of missing time-gaps**: Enhance the methods used to fill in these gaps.
4. **Experiment with further anomaly detection methods**: Explore methods such as ARIMA and VAEs for improved results.
5. **Conduct extensive validation of detected anomalies**: Ensure the robustness and accuracy of the anomaly detection process through thorough validation.
6. **Documentation**: Some parts of the code could gain more clarity by providing inline-documentation, e.g., the ```src/window_generator.py``` file.
