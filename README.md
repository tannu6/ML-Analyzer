# 🚀 SmartML Analyzer

SmartML Analyzer is a simple and interactive machine learning web app built using Streamlit. It allows users to upload any CSV dataset, train multiple ML models, visualize data, and evaluate model performance easily — all in one place.

---

## 📌 Features

* 📂 Upload any CSV dataset
* 🤖 Train multiple machine learning models:

  * Linear Regression
  * Multiple Linear Regression
  * Polynomial Regression (auto best degree)
  * KNN Classifier (auto best K)
  * Decision Tree Classifier (auto best depth)
* 📊 Performance Metrics:

  * Accuracy, Recall, F1 Score
  * R² Score (for regression)
* 📉 Confusion Matrix
* 📊 Data Visualization:

  * Bar Chart
  * Line Chart
  * Scatter Plot
  * Pie Chart
* 🔥 Correlation Heatmap
* 🧹 Automatic handling of missing values

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
---

## ▶️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/smartml-analyzer.git
cd smartml-analyzer
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run Model_training.py
```

4. Open in browser:

```
http://localhost:8501
```

---

## 💡 How It Works

1. Upload your CSV dataset
2. Select the target column
3. Choose a machine learning model
4. View performance metrics
5. Explore data with visualizations
6. Analyze relationships using heatmap

---

## 📊 Example Use Cases

* Beginner ML projects
* Data analysis practice
* Quick model comparison
* Educational demos

---

## ⚠️ Limitations

* Works best with clean datasets
* No advanced preprocessing (scaling, encoding customization)
* KNN and Decision Tree assume classification tasks

---

## 🚀 Future Improvements

* 🤖 Auto best model selection
* 📊 Model comparison dashboard
* 🌐 Cloud deployment (Streamlit Cloud)
* 🔍 Feature importance visualization
* ⚙️ Advanced preprocessing options

---

## 👨‍💻 Author

Tannu Sharma

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
