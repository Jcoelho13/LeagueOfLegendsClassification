# Group A2_20 - Assignment 2: Classification

## Group members

- João Coelho (up202004846@up.pt)
- João Mota (up202108677@up.pt)
- Pedro Landolt (up202103337@up.pt)

## Repository organization

- `data/` - folder with the dataset file
- `models/` - folder with the trained models
- `predict_data/` - folder to place files to make predictions on
- `matrix/` - folder with the confusion matrices of the models
- `main.py` - main script to run the project

## Instructions

### Requirements:
- scikit-learn: version 1.5.0 is needed, otherwise warnings will be raised
- pandas
- numpy
- matplotlib
- pickle

We recommend using conda to install the required packages.

PyCharm is also recommended to run the project, as it simplifies the visualization of the plots.

### Running the project

3 ways to run the project:
- Run the `main.py` script in the root of the project, using PyCharm - **recommended**
- Run the `main.py` script in the root of the project, using coderunner in VSCode
- Run the command `<path to interpreter> <path to main.py>` in the root of the project, on a terminal:

Example:
```bash
/home/jcoelho13/anaconda3/bin/python /home/jcoelho13/Desktop/leic/y3/s2/ia/IA-LeagueOfLegends/main.py
```

### How to use the project
Able to:
- Print the dataset
- Visualize statistics of the dataset, such as average values of features, check for missing values, view every point in a scatter plot
- Create models, from 7 different types of classifiers, with and without the usage of PCA
- See data about the models: accuracy, precision, recall, f1-score, confusion matrix
- Use created models and pre-existing models to predict the outcome of a file, or a single sample via the terminal

Simply run the project and follow the instructions on the terminal.
