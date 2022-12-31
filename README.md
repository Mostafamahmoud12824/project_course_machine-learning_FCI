# project_course_machine-learning_FCI
# In this project Given the attached dataset for regression problem with one input and one output, do the following:
-----------------------
* LOAD regression_data.csv to dataframe pandas ( the first column is input x and the second column is the output)
* plot the data as scatterplot
* As described in the lectures, use the linear regression model, and Root Mean Square as cost function
* build matrix X with the input data (do not forget that the x0 = 1, this not included in the data)
* initialize the theta vector with zeros
* learning rate is 0.01
* number of epochs 1500
* define function that compute J as cost value (Root Mean Square as cost function)
* Report the cost when theta is zeros
* As described in the lecture, Define a function that compute the batch gradient descent.
* After iterate using 1500 iteration, report the final theta you have
* Plot the data as scatter and the model as line.
* Given input = [1, 3.5], what is the predicted output from your model?
* plot a contour-plot given your thetas and the cost function J
* provide a written report with the equations you implement with descriptions for each one, your results, and the figures. Also, you will deliver the code and it should run with correct results.
* The code should has clear documentations for each variable and its connection to the equations you will implement.
* The code should be able to run without any changing for any number of features (so, if we change the data to have 3 features as input for example, all computations should work without any changes). Without writing these functions in general form, you lose points.
* Reducing Loss: Optimizing Learning Rate
