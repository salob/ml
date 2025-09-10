# ml

## Tools

Python: The primary programming language used for its ease of learning and extensive libraries.

Jupyter Notebooks: An integrated development environment for writing Python code, data analysis, and visualization.

Pandas: A library for working with data structures and performing exploratory data analysis.

NumPy: A library for numerical computing, supporting multidimensional arrays and matrices.

Matplotlib and Seaborn: Libraries for creating 2D plots and visualizations.

Scikit-Learn: A machine learning framework used for training custom models, known for its ease of use and extensive documentation.


## Scenarios

### Predicting Home Prices in California

Uses supervised learning to solve a linear regression problem by predicting home prices based on features like location, age, number of bedrooms, and ocean proximity.

### Classifying Flowers in Images

Uses supervised learning to solve a multi-label image classification problem by identifying different types of flowers from images.

### Predicting Crime in the UK

Uses supervised learning to solve a binary classification problem by predicting whether a stop and search will lead to an arrest based on features like location, gender, age, ethnicity, and time of day.

## Definition of terms

### AI and Machine Learning

AI simulates human intelligence, and machine learning is a subset of AI that uses mathematics to find patterns in large data sets and map inputs to outputs.
    
### Machine Learning Models

These models store the mappings between inputs and outputs and are often depicted as a brain to simulate human intelligence.
    
### Resurgence of Machine Learning

The excitement around machine learning is due to the availability of big data and cloud computing, which provide the necessary data and computational power.

### Daily Impact
Machine learning affects daily life in areas like streaming recommendations, loan approvals, fraud detection, and customer loyalty programs.

### Mathematical Foundation

Basic knowledge of linear algebra, equations, probability, and programming (especially Python) is helpful but not mandatory, as widely used libraries can assist with the heavy lifting.

## Machine Learning Techniques

### Supervised Learning

Trains a model with labeled data to predict future outputs. Examples include predicting home prices (regression) and classifying flowers (classification).

### Unsupervised Learning

Uses input data without labels to group or interpret data. A common technique is clustering, used in the hospitality industry for customer segmentation.

### Reinforcement Learning

Allows machines to learn through trial and error in a dynamic environment, similar to training a dog with rewards. It's used in robotics and AI programs that play board games like Go and chess.

## Machine Learning Lifecycle


### Problem formation and understanding

Is ML an ethical solution for this problem, define inputs/outputs and error acceptance rate. Identifying where machine learning adds value and defining inputs, outputs, and acceptable prediction error rates.

### Data collection and preparation

Sourcing and preparing data, including labeling, removing irrelevant features, and splitting data into training 80%, validation 10%, and testing 10% sets.

Most time spent in this stage.

### Model training and testing

Select machine learning algorithm, training the model, and iterating to fine-tune and evaluate performance.

### Model deployment and maintenance.

Model Deployment and Maintenance: Deploying the model and setting up a cadence for retraining and monitoring performance.

## Framing Machine Learning Problems

Not the solution to every problem and when used incorrectly can do more damage than good

If rules, computation, pre-determined steps can be used to solve problem machine learning is not the solution

Works best when:

- you need to predict an outcome

- uncover trends and patterns in data

- have rules and pre-determined steps that can't be coded

- dataset is too large to be processed by a human

Framing also helps answer things like selecting the appropriate algorith e.g. for binary classification (yes/no) answers you could use AWS linear learner. If expecting numeric or continuous value outputs XGBoost algorithm could be the one.

Also what is the success criteria e.g. on a real estate website maybe it could be sales increased by 20%

### Importance of Framing

Correctly framing your machine learning problem is crucial for project success, helping to determine feasibility, goals, and success criteria.

### Defining Questions and Inputs

Start by defining the questions for the model, required inputs, and expected outputs. This helps in selecting the appropriate algorithm.

### Data Requirements

Ensure you have the necessary data to train the model and understand the relationships between features and the target.

### Integration and Metrics

Consider how the model's output will integrate with existing applications and define success metrics upfront to align team goals.

## Prebuilt Models

These are models already trained on large benchmark datasets to solve problems similar to yours, saving time and resources.


### Transfer Learning

This technique allows you to add your data on top of a pre-trained model, leveraging previous learnings for efficiency and reusability.

### Sources for Pre-trained Models

- ModelZoo: A popular catalog for various frameworks and domains.
- AWS Marketplace: Offers ready-to-use models that can be deployed on Amazon SageMaker.
- Hugging Face: Provides pre-trained models for natural language processing tasks like sentiment analysis and language translation.

## Training Custom Models

Simple models can be trained on your local machine using the normal CPU that every laptop has.

For more complex training that requires significant computational power, such as using GPUs, cloud providers like Amazon Web Services (AWS) are recommended. AWS provides easy access to GPUs, which are necessary for handling large datasets and complex models.

### Data and Hardware

Data is crucial for training a custom model. Simple models can be trained on a local machine, but complex models often require GPUs, which can be accessed through cloud providers like AWS.

### Programming Languages

Python is recommended due to its ease of learning and extensive libraries for machine learning and data analysis.

### Tools and Libraries

Key tools include Jupyter Notebooks for development, Pandas and NumPy for data analysis, Matplotlib and Seaborn for data visualization, and Scikit-Learn for training models.

## Obtaining Data


Data is a critical element of any machine learning project. Data can be sourced from internal data stores, clients, open-source or public data stores, or purchased from third parties.

Example Project: Housing price prediction model as an example, explaining how various features such as demography, location, and house characteristics are used in a supervised learning model to predict house prices.

Linear regression is a supervised learning algorithm where machines are trained using labeled data. It predicts a numeric value (target) based on the relationships between independent variables (features). In the example project, linear regression is used to predict the median house value based on features like location, number of rooms, and median income.
