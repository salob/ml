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

# Code Carbon project

First test

```
[codecarbon INFO @ 19:28:56] Codecarbon is taking the configuration from the local file /Users/sianob/git/ml/.codecarbon.config
[codecarbon WARNING @ 19:28:56] Multiple instances of codecarbon are allowed to run at the same time.
[codecarbon WARNING @ 19:28:56] Error while trying to count physical CPUs: [Errno 2] No such file or directory: 'lscpu'. Defaulting to 1.
[codecarbon INFO @ 19:28:56] [setup] RAM Tracking...
[codecarbon INFO @ 19:28:56] [setup] CPU Tracking...
[codecarbon WARNING @ 19:28:56] We saw that you have a Apple M3 but we don't know it. Please contact us.
Password:
[codecarbon INFO @ 19:29:02] Tracking Apple CPU and GPU via PowerMetrics
[codecarbon INFO @ 19:29:02] [setup] GPU Tracking...
[codecarbon INFO @ 19:29:02] No GPU found.
[codecarbon INFO @ 19:29:02] The below tracking methods have been set up:
                RAM Tracking Method: RAM power estimation model
                CPU Tracking Method: PowerMetrics
                GPU Tracking Method: PowerMetrics
            
[codecarbon INFO @ 19:29:02] >>> Tracker's metadata:
[codecarbon INFO @ 19:29:02]   Platform system: macOS-15.6.1-arm64-arm-64bit-Mach-O
[codecarbon INFO @ 19:29:02]   Python version: 3.13.3
[codecarbon INFO @ 19:29:02]   CodeCarbon version: 3.0.4
[codecarbon INFO @ 19:29:02]   Available RAM : 16.000 GB
[codecarbon INFO @ 19:29:02]   CPU count: 8 thread(s) in 1 physical CPU(s)
[codecarbon INFO @ 19:29:02]   CPU model: Apple M3
[codecarbon INFO @ 19:29:02]   GPU count: 1
[codecarbon INFO @ 19:29:02]   GPU model: Apple M3
[codecarbon INFO @ 19:29:05] Emissions data (if any) will be saved to file /Users/sianob/git/ml/emissions.csv
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   MedInc       20640 non-null  float64
 1   HouseAge     20640 non-null  float64
 2   AveRooms     20640 non-null  float64
 3   AveBedrms    20640 non-null  float64
 4   Population   20640 non-null  float64
 5   AveOccup     20640 non-null  float64
 6   Latitude     20640 non-null  float64
 7   Longitude    20640 non-null  float64
 8   MedHouseVal  20640 non-null  float64
dtypes: float64(9)
memory usage: 1.4 MB
[codecarbon INFO @ 19:29:17] Energy consumed for RAM : 0.000010 kWh. RAM Power : 3.0 W
[codecarbon INFO @ 19:29:18] Energy consumed for all CPUs : 0.000001 kWh. Total CPU Power : 0.21990000000000004 W
[codecarbon INFO @ 19:29:20] Energy consumed for all GPUs : 0.000000 kWh. Total GPU Power : 0.0087 W
[codecarbon INFO @ 19:29:20] 0.000011 kWh of electricity used since the beginning.
```

## Challenges

data preparation challenges different models expect different data structures for input

mac silicon and access to live power consumption metrics
https://github.com/vladkens/macmon

## ml tutorials

https://medium.com/@gourish.deshpande/training-the-tiny-transformer-properly-7dfafb712f9a


## ideas / snippets

For this study I use the IMDB film review dataset to train and evaluate models on a binary classification task: predicting review sentiment (positive vs. negative).

what is an epoch (pronounced epic)

An epoch is one full pass through the entire training dataset.

Think of it like reading a whole stack of flashcards once: the model looks at every example one time, makes predictions, sees how wrong it is, and updates itself a bit. Multiple epochs mean repeating that process many times so the model gradually improves.

the model will converge over time i.e. it might not get better after the 10th epoch. 10 epochs was chosen for cnn as that is when convergence was noted and 

picking the right epoch budget:

There were two options: 

1. measure everything (training + validation) for every model
2. Run a small untracked pilot to pick a single epoch budget for the deep learning models and then run the final fully tracked experiments with that budget.

1. is still fair and simple and tracks the full workflow for each model (include validation cost). This is most honest because validation is real work and CodeCarbon will count it equally for all models.

2. I could use a small subset to find a convergence epoch, then run final tracked experiments once per model using that chosen epoch e.g. i'm using 10 right now. I would just need to document that pilot runs were excluded from tracking just to make it fairer compared to the less intensive logistic regression model.

I could also do a mix or hybrid approach and only validate every N epochs or validate on a small validation subset to reduce overhead while still monitoring convergence.

what about the built in AI forecasting, how good is an ai model at forecasting ?:
That is what the loss function is for it can track the loss between a predicted value and the actual value. If predictions are accurate then the loss or difference between these numbers is small.

training uses randomness for things like batch shuffling and weight initialisation. You can make it more deterministic by using the same seed and deterministic algorithms from the torch library. However complete determination can be slow or impossible across platforms and versions so a better approach is to run multiple times and take a mean performance score. The same will be done for all models.