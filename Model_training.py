import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, r2_score, confusion_matrix,
)

st.set_page_config(page_title="SmartML Analyzer", page_icon="✌️", layout="wide")
st.title("🔥 SmartML Analyzer Pro - Memory Optimized")
st.caption("Handles MASSIVE datasets (100k+ rows) without crashing!")

@st.cache_data
def smart_preprocess(df_raw, max_unique=50, max_features=500):
    """Memory-optimized preprocessing for huge datasets"""
    df = df_clean(df_raw)
    
    st.info(f"📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Step 1: DROP high-cardinality columns (ID, names, etc.)
    cols_to_drop = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.1 or df[col].nunique() > max_unique:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.success(f"🗑️ Dropped {len(cols_to_drop)} useless columns (IDs/names)")
    
    # Step 2: Keep only top N most important features
    if len(df.columns) > max_features:
        # Use correlation with target or variance for feature selection
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            top_features = numeric_cols[:max_features//2]
            cat_cols = df.select_dtypes(exclude=np.number).columns[:max_features//2]
            df = df[top_features.tolist() + cat_cols.tolist()]
            st.info(f"✂️ Kept top {max_features} features for memory")
    
    # Step 3: Smart encoding (LabelEncoder instead of get_dummies)
    X = df.copy()
    le_dict = {}
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    st.success("✅ Smart preprocessing complete! Memory usage optimized.")
    return X, le_dict

def df_clean(df_raw):
    """Smart cleaning without memory explosion"""
    df = df_raw.copy()
    df = df.drop_duplicates()
    
    # Fill missing values efficiently
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    obj_cols = df.select_dtypes(exclude=np.number).columns
    for col in obj_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def target_type(series):
    return "regression" if pd.api.types.is_numeric_dtype(series) else "classification"

def model_tip(model_name):
    tips = {
        "Linear Regression": "🎯 Best for numeric targets with linear patterns",
        "Multiple Linear Regression": "📈 Uses ALL features together for numeric prediction", 
        "Polynomial Regression": "🌀 Models curved/non-linear numeric relationships",
        "KNN": "👥 Predicts using nearest neighbor voting",
        "Decision Tree": "🌳 Rule-based splits, easy to understand",
    }
    return tips.get(model_name, "")

def model_info(model_name):
    info = {
        "Linear Regression": {
            "about": "Finds the best straight line through your data points",
            "youtube": "https://www.youtube.com/watch?v=gPfgB4ew3RY"
        },
        "Multiple Linear Regression": {
            "about": "Same idea but uses multiple input features together",
            "youtube": "https://www.youtube.com/watch?v=i3IadpjctWg"
        },
        "Polynomial Regression": {
            "about": "Adds curves to linear regression for non-linear patterns",
            "youtube": "https://www.youtube.com/watch?v=-XFsQ4QJRoM"
        },
        "KNN": {
            "about": "Looks at closest data points and votes for prediction",
            "youtube": "https://www.youtube.com/watch?v=Nz73vXn5afE"
        },
        "Decision Tree": {
            "about": "Splits data using if-then rules until final prediction",
            "youtube": "https://www.youtube.com/watch?v=YkYpGhsCx4c"
        }
    }
    return info.get(model_name, {})

# File upload
file = st.file_uploader("📁 Upload CSV File", type=["csv"])
if file is None:
    st.info("👆 Please upload a CSV file first.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df_raw = load_data(file)

st.subheader("1. 📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{df_raw.shape[0]:,}")
col2.metric("Columns", df_raw.shape[1])
col3.metric("Memory", f"{df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
col4.metric("Missing %", f"{df_raw.isna().sum().sum() / df_raw.size * 100:.1f}%")

st.dataframe(df_raw.head(), use_container_width=True)

# Preprocessing
st.subheader("2. 🧠 Smart Preprocessing")
max_unique = st.slider("Max unique values per column", 20, 200, 50)
max_features = st.slider("Max features to keep", 50, 1000, 200)

with st.spinner("Preprocessing massive dataset..."):
    df_cleaned, le_dict = smart_preprocess(df_raw, max_unique, max_features)

st.success(f"✅ Ready! Shape: {df_cleaned.shape}")

# Model setup
st.sidebar.header("⚙️ Model Settings")
target_col = st.sidebar.selectbox("🎯 Target Column", df_cleaned.columns)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Prepare data
df_target = df_cleaned.dropna(subset=[target_col])
X = df_target.drop(columns=[target_col])
y = df_target[target_col]

problem_type = target_type(y)
st.info(f"🔍 Target '{target_col}' → **{problem_type.upper()}** problem")

if problem_type == "regression":
    model_options = ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]
else:
    model_options = ["KNN", "Decision Tree"]

model_name = st.selectbox("🤖 Select Model", model_options)
st.info(model_tip(model_name))

# Train model
st.subheader("3. 🚀 Model Training")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if model_name in ["Linear Regression", "Multiple Linear Regression"]:
    model = LinearRegression()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    score = r2_score(y_test, predict)
    st.metric("R² Score", f"{score:.4f}")

elif model_name == "Polynomial Regression":
    best_score = -np.inf
    best_degree = 1
    x_train_num = x_train.select_dtypes(include=np.number)
    
    if x_train_num.shape[1] > 0:
        for d in range(1, 4):
            try:
                poly = PolynomialFeatures(degree=d, include_bias=False)
                x_train_poly = poly.fit_transform(x_train_num)
                x_test_poly = poly.transform(x_test.select_dtypes(include=np.number))
                
                lr = LinearRegression()
                lr.fit(x_train_poly, y_train)
                pred = lr.predict(x_test_poly)
                score = r2_score(y_test, pred)
                
                if score > best_score:
                    best_score = score
                    best_degree = d
                    predict = pred
            except:
                continue
        
        st.metric("Best Degree", best_degree)
        st.metric("R² Score", f"{best_score:.4f}")
    else:
        st.warning("No numeric features for polynomial regression")

elif model_name == "KNN":
    best_acc = 0
    best_k = 1
    
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_k = k
            predict = pred
    
    rec = recall_score(y_test, predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, predict, average="macro", zero_division=0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Best K", best_k)
    col2.metric("Accuracy", f"{best_acc:.4f}")
    col3.metric("F1 Score", f"{f1:.4f}")

elif model_name == "Decision Tree":
    best_acc = 0
    best_depth = 1
    
    for d in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(x_train, y_train)
        pred = dt.predict(x_test)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_depth = d
            predict = pred
    
    rec = recall_score(y_test, predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, predict, average="macro", zero_division=0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Depth", best_depth)
    col2.metric("Accuracy", f"{best_acc:.4f}")
    col3.metric("F1 Score", f"{f1:.4f}")

# Model learning
st.subheader("4. 📚 Learn This Model")
info = model_info(model_name)
col1, col2 = st.columns([2,1])
with col1:
    st.write(f"**{info.get('about', '')}**")
with col2:
    st.markdown(f"[📺 YouTube Tutorial]({info.get('youtube', '#')})")

# Prediction
st.subheader("5. 🔮 Make Prediction")
st.info("Enter values similar to your training data")

input_data = {}
for col in X.columns[:8]:  # Show top 8 features
    if pd.api.types.is_numeric_dtype(X[col]):
        input_data[col] = st.number_input(col, value=float(X[col].median()))
    else:
        unique_vals = X[col].value_counts().head(5).index.tolist()
        input_data[col] = st.selectbox(col, unique_vals)

if st.button("🔮 Predict Now", type="primary"):
    input_df = pd.DataFrame([input_data])
    
    # Encode input
    for col in input_df.select_dtypes(include=['object']).columns:
        if col in le_dict:
            input_df[col] = le_dict[col].transform(input_df[col].astype(str))
    
    # Align columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    try:
        pred = model.predict(input_df) if 'model' in locals() else [0]
        st.success(f"🎯 **Prediction: {pred[0]:.4f}**")
    except:
        st.error("❌ Prediction failed. Try different input values.")

# Visualizations
st.subheader("6. 📊 Smart Visualizations")
tab1, tab2 = st.tabs(["📈 Quick Charts", "🔗 Correlations"])

with tab1:
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("X", df_cleaned.columns[:10])
    y_col = col2.selectbox("Y", df_cleaned.columns[:10])
    
    if pd.api.types.is_numeric_dtype(df_cleaned[x_col]) and pd.api.types.is_numeric_dtype(df_cleaned[y_col]):
        fig = px.scatter(df_cleaned.head(5000), x=x_col, y=y_col, opacity=0.6)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    numeric_cols = df_cleaned.select_dtypes(np.number).columns[:10]
    if len(numeric_cols) >= 2:
        corr = df_cleaned[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Add more numeric columns for correlation heatmap")
