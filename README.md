## Code Structure and Organization

For this project, the code is organized in a way that makes it easy to understand, maintain, and extend. The overall structure is as follows:

1. **Main File (`main.jl`)**: This file contains the core logic of the project. It is responsible for loading the dataset, preprocessing the data, training the models, and evaluating the results. The main file calls functions from the `utils.jl` file to keep everything modular and organized.

2. **Utils File (`utils.jl`)**: This file contains all the necessary helper functions that are needed throughout the project. It includes functions for tasks such as data preprocessing, model evaluation, and generating confusion matrices. These functions are reusable and allow for better code organization.

3. **Datasets Folder**: This folder contains the CSV file(s) that are used throughout the project. The data is used for all the models and approaches, ensuring consistency across the experiments.

4. **Images Folder**: This folder contains all the graphs and plots generated during the project. Visualizations, such as model performance metrics, confusion matrices, and other relevant charts, are saved in this folder. These images are typically used for reporting, analysis, or as a reference for comparing different models and results. All graphs are stored here, and they are referenced in the report or notebook.

5. **Main Output File (`main_output.txt`)**: This file stores the outputs of the project, including model evaluation results, performance metrics, and any console outputs generated during the execution of the project. It serves as a log file, capturing important results and information for further analysis or review.

6. **Main Notebook (`main.ipynb`)**: This Jupyter notebook serves as an interactive version of the core logic in the project. It is used for experimentation, visualization, and step-by-step explanation of the model building, training, and evaluation process. The notebook provides a more user-friendly interface for exploring the project in an interactive manner.

7. **HTML File (`report.html`)**: This file is the HTML conversion of the `main.ipynb` notebook. It contains a static, formatted version of the notebook, including all code, outputs, and visualizations. The HTML report is useful for sharing the results in a browser-readable format without requiring users to run the code.

This structure ensures that the project is easy to maintain and understand, with each component clearly defined. The inclusion of the `main_output.txt` file allows for detailed logging of outputs, and the `images` folder ensures that all visualizations are well-organized and easy to access.
