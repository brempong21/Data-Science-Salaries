# Data Science Salaries Dataset

In the ever-evolving field of data science, staying abreast of salary trends is crucial for both professionals and organizations. This dataset provides a comprehensive view of Data Science Salaries spanning the years 2020 to 2024. It serves as a valuable resource for data enthusiasts, researchers, and industry professionals seeking nuanced insights into compensation dynamics.

## Overview

Covering a five-year period, this dataset offers a detailed collection of data science salary information. It explores various facets of salaries, presenting a nuanced perspective on compensation trends within the field.

### Columns

1. **Job Title:** The professional title of the individual.
2. **Experience Level:** The level of professional experience.
3. **Employment Type:** The nature of employment (e.g., full-time, part-time, contract).
4. **Work Models:** Models or methodologies utilized in the work.
5. **Work Year:** The specific year to which the salary information corresponds.
6. **Employee Residence:** The location of the employee's residence.
7. **Salary:** Raw salary data.
8. **Salary Currency:** The currency in which the salary is reported.
9. **Salary in USD:** Standardized salary in US dollars.
10. **Company Location:** The location of the employing company.
11. **Company Size:** Size category of the employing company.

## Objective

This dataset facilitates a comprehensive analysis of salary trends, regional disparities, and potential factors influencing compensation within the data science community. It serves as a robust foundation for exploratory data analysis and further research in the realm of data science salaries.

# Data Preprocessing

During the preprocessing phase, we encoded all categorical columns to facilitate model training. Here are the categorical columns after encoding:

```python
Categorical Columns:
Index(['job_title', 'experience_level', 'employment_type', 'work_models',
       'employee_residence', 'salary_currency', 'company_location',
       'company_size'],
      dtype='object')
```

The categorical variables was encoded during the preprocessing phase and saved the processed data as a new CSV file. You can access the encoded dataset using the following link:

[Encoded Dataset](https://raw.githubusercontent.com/brempong21/Data-Science-Salaries/main/encoded.csv)



# Regression Models Performance

After evaluation, the performance of various regression models on our dataset using R-squared (R2) score as the metric. Below are the results:

## Models

```python
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}

Model Performance
Linear Regression:
R2 Score: 0.2

Ridge Regression:
R2 Score: 0.2

Lasso Regression:
R2 Score: 0.2

Random Forest:
R2 Score: 0.96

Gradient Boosting:
R2 Score: 0.99

Decision Tree:
R2 Score: 0.9
```

# Model Performance

After thorough evaluation of various regression models on our dataset, the Gradient Boosting model emerged as the top performer with an impressive R-squared (R2) score of 0.99. This signifies a high level of accuracy and effectiveness in predicting data science salaries.

## Best Performing Model: Gradient Boosting

- **Gradient Boosting:**
  - R2 Score: 0.99

This outstanding performance makes the Gradient Boosting model a compelling choice for predicting and understanding data science compensation trends in our dataset. 

