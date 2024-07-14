# Product Category Prediction

This project predicts the category of a product based on its description. The application is built using Flask for the web interface and various machine learning models for the prediction.

## Project Structure

- `src/`: Contains the source code for the Flask application and inference logic.
- `data/`: Contains the raw data files used for training.
- `model/`: Contains the trained models and vectorizers.
- `preprocessed/`: Contains the preprocessed data.
- `requirements.txt`: Lists the dependencies required for the project.
- `Dockerfile`: Dockerfile to containerize the application.
- `docker-compose.yml`: Docker Compose file to manage the Docker container.
- `task_main.ipynb`: Jupyter Notebook file that includes report
- There are preprocessed.pkl files besides files in directory `preprocessed`. These are created for `task_main.ipynb`

## Setup

### Prerequisites

- Docker and Docker Compose installed on your machine.

### Running the Project

1. Build the Docker image:

```bash
docker-compose build
```
2. Start the Docker container:

```bash
docker-compose up
```
3. Access the application at http://127.0.0.1:8000

### Using the Application

Enter a product description in the text area and click "Predict" to get the predicted category.

### Project Components

# Data Preprocessing
Load and clean the data.
Vectorize the text descriptions.
Handle class imbalance using SMOTE.

# Model Training
Train multiple models (Logistic Regression, SVM, Random Forest, Decision Tree, Naive Bayes, K-Nearest Neighbors).
Perform hyperparameter tuning using GridSearchCV.
Save the best model and vectorizer.

# Inference
Load the best model and vectorizer.
Predict the category for new product descriptions.

# Conclusion

This project demonstrates the process of building a machine learning model to predict product categories from descriptions. It includes data preprocessing, model training and evaluation, and deployment using Docker and Flask.

### Önemli not
- Modelleri başarılı bir şekilde eğitip raporlarını çıkarabildim. Ama hem bu task'a geç başladığım (sürece sonradan dahil oldum) hem de hafta içlerinde bir şirkette staj yaptığım için yeterli zaman bulamadım. Bu yüzden eğittiğim modelleri bir türlü uygulamaya düzgün entegre edemedim.
