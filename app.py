from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from preprocess import check_data_quality, handle_preprocessing_action
from visualize import get_columns, generate_plot

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans

# Auth helpers
from auth import (
    init_db, create_user, authenticate_user,
    email_exists, create_reset_token, validate_reset_token,
    mark_token_used, update_password, send_reset_email
)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# Initialise DB on startup
init_db()

trained_models = {}


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def read_rows_from_csv(filepath, start, end):
    chunk_size = end - start
    skip_rows  = list(range(1, start + 1))
    try:
        df = pd.read_csv(filepath, skiprows=skip_rows, nrows=chunk_size)
        df.columns = pd.read_csv(filepath, nrows=0).columns
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


# ─────────────────────────────────────────────
# Auth routes
# ─────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        user, err = authenticate_user(email, password)
        if user:
            session['logged_in']   = True
            session['user_email']  = user['email']
            session['user_name']   = user['firstname']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=err, active_tab='login')

    return render_template('login.html', active_tab='login')


@app.route('/signup', methods=['POST'])
def signup():
    firstname = request.form.get('firstname', '').strip()
    email     = request.form.get('email', '').strip()
    password  = request.form.get('password', '').strip()

    if not firstname or not email or not password:
        return render_template('login.html', error='Please fill in all fields.', active_tab='signup')

    if len(password) < 6:
        return render_template('login.html', error='Password must be at least 6 characters.', active_tab='signup')

    success, err = create_user(firstname, email, password)
    if success:
        return render_template('login.html',
                               success='Account created! You can now log in.',
                               active_tab='login')
    else:
        return render_template('login.html', error=err, active_tab='signup')


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form.get('email', '').strip()

    if not email:
        return render_template('login.html', error='Please enter your email address.', active_tab='forgot')

    # Always show the same message to prevent email enumeration
    if email_exists(email):
        token    = create_reset_token(email)
        base_url = request.host_url.rstrip('/')
        send_reset_email(email, token, base_url)

    return render_template(
        'login.html',
        success='If that email is registered, a reset link has been sent. Check your inbox.',
        active_tab='forgot'
    )


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email, err = validate_reset_token(token)

    if err:
        return render_template('login.html', error=err, active_tab='forgot')

    if request.method == 'POST':
        new_password     = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not new_password or len(new_password) < 6:
            return render_template('reset_password.html', token=token, error='Password must be at least 6 characters.')

        if new_password != confirm_password:
            return render_template('reset_password.html', token=token, error='Passwords do not match.')

        update_password(email, new_password)
        mark_token_used(token)

        return render_template('login.html',
                               success='Password updated successfully! You can now log in.',
                               active_tab='login')

    return render_template('reset_password.html', token=token, error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ─────────────────────────────────────────────
# Main app routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file uploaded", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        session['filepath'] = filepath
        session.pop('target_col', None)
        return redirect(url_for('dataset_preview'))

    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    try:
        start = int(request.args.get('start', 0))
        end   = int(request.args.get('end', start + 5))
    except ValueError:
        return "Invalid row range", 400

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading file: {e}", 500

    sort_by = request.args.get('sort_by')
    order   = request.args.get('order', 'asc')
    if sort_by and sort_by in df.columns:
        df.sort_values(by=sort_by, ascending=(order == 'asc'), inplace=True)

    df_range       = df.iloc[start:end]
    column_names   = df.columns.tolist()
    row_data       = df_range.to_dict(orient='records')
    total_rows     = len(df)
    total_columns  = len(df.columns)
    missing_percent = round(df.isnull().sum().sum() / (total_rows * total_columns) * 100, 1)
    duplicate_rows  = df.duplicated().sum()

    return render_template("preview.html",
                           filename=os.path.basename(filepath),
                           column_names=column_names,
                           row_data=row_data,
                           total_rows=total_rows,
                           total_columns=total_columns,
                           missing_percent=missing_percent,
                           duplicate_rows=duplicate_rows,
                           start=start,
                           end=end,
                           sort_by=sort_by,
                           order=order)


@app.route('/preprocess', methods=['GET'])
def preprocess():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    missing_info, duplicate_count, columns = check_data_quality(filepath)
    filename = os.path.basename(filepath)

    return render_template('preprocess.html',
                           missing_info=missing_info,
                           duplicate_count=duplicate_count,
                           columns=columns,
                           filename=filename)


@app.route('/preprocess_action', methods=['POST'])
def preprocess_action():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    action = request.form.get("action")
    if handle_preprocessing_action(filepath, action, request.form):
        return redirect(url_for('dataset_preview'))
    else:
        return "Error performing preprocessing action", 500


@app.route('/preview')
def dataset_preview():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    try:
        start = int(request.args.get('start', 0))
        end   = int(request.args.get('end', start + 100))
    except ValueError:
        return "Invalid row range", 400

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading file: {e}", 500

    sort_by = request.args.get('sort_by')
    order   = request.args.get('order', 'asc')
    if sort_by and sort_by in df.columns:
        df.sort_values(by=sort_by, ascending=(order == 'asc'), inplace=True)

    df_range        = df.iloc[start:end]
    column_names    = df.columns.tolist()
    row_data        = df_range.to_dict(orient='records')
    total_rows      = len(df)
    total_columns   = len(df.columns)
    missing_percent = round(df.isnull().sum().sum() / (total_rows * total_columns) * 100, 1)
    duplicate_rows  = df.duplicated().sum()

    return render_template("preview.html",
                           filename=os.path.basename(filepath),
                           column_names=column_names,
                           row_data=row_data,
                           total_rows=total_rows,
                           total_columns=total_columns,
                           missing_percent=missing_percent,
                           duplicate_rows=duplicate_rows,
                           start=start,
                           end=end,
                           sort_by=sort_by,
                           order=order)


@app.route('/data_quality')
def data_quality():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    quality_report = {}

    for col in df.columns:
        stats = {
            'missing': int(df[col].isnull().sum()),
            'unique':  int(df[col].nunique()),
            'mean':    round(df[col].mean(), 2) if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
            'mode':    df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
            'sd':      round(df[col].std(), 2)  if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
            'q1':      round(df[col].quantile(0.25), 2) if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
            'q3':      round(df[col].quantile(0.75), 2) if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
            'iqr':     round(df[col].quantile(0.75) - df[col].quantile(0.25), 2) if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        }
        quality_report[col] = stats

    return render_template('data_quality.html',
                           filename=os.path.basename(filepath),
                           quality_report=quality_report)


@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    numeric_cols, cat_cols, df = get_columns(filepath)
    filename = os.path.basename(filepath)
    plot_filename       = None
    selected_chart_type = None

    if request.method == 'POST':
        chart_type = request.form.get('chart_type', 'bar')
        x_axis     = request.form.get('x_axis')
        y_axis     = request.form.get('y_axis')
        z_axis     = request.form.get('z_axis')

        selected_chart_type = chart_type
        plot_filename, _ = generate_plot(df, chart_type, x_axis, y_axis, z_axis)

    return render_template('visualize.html',
                           filename=filename,
                           numeric_cols=numeric_cols,
                           cat_cols=cat_cols,
                           plot_filename=plot_filename,
                           selected_chart_type=selected_chart_type)


# ─────────────────────────────────────────────
# ML helpers
# ─────────────────────────────────────────────

def train_supervised_models(df, target_col, feature_cols):
    df = df.copy()
    df.dropna(subset=[target_col] + feature_cols, inplace=True)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    cat_unique_values = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            cat_unique_values[col] = X[col].dropna().unique().tolist()

    X = pd.get_dummies(X)
    feature_names = X.columns

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":          LinearRegression(),
        "Decision Tree Regressor":    DecisionTreeRegressor(),
        "Random Forest Regressor":    RandomForestRegressor(n_estimators=200),
        "SVR":                        SVR(),
        "KNN Regressor":              KNeighborsRegressor()
    }

    results = []
    trained_models.clear()

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results.append({"model": name, "r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)})
        trained_models[name] = (model, feature_names, scaler)

    # Model comparison chart
    model_names_list = [r['model'] for r in results]
    r2_list          = [r['r2'] for r in results]
    short_names      = [n.replace(' Regressor','').replace(' Regression','') for n in model_names_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(short_names, r2_list, color='#1f77b4', edgecolor='none', width=0.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    for bar, val in zip(bars, r2_list):
        ypos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.012
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Model Comparison', fontsize=13)
    ax.set_ylabel('R2 Score')
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig('static/plots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance chart
    rf_entry = trained_models.get("Random Forest Regressor")
    if rf_entry:
        rf_model, feat_names, _ = rf_entry
        importances = rf_model.feature_importances_
        feat_series = pd.Series(importances, index=feat_names).sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(8, max(4, len(feat_series) * 0.4)))
        ax.barh(feat_series.index, feat_series.values, color='#1f77b4', edgecolor='none')
        ax.set_title('Feature Importance', fontsize=12)
        ax.set_xlabel('Importance Score')
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig('static/plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    return results, list(feature_names), cat_unique_values


@app.route('/ml_models', methods=['GET', 'POST'])
def ml_models():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    columns = df.columns.tolist()

    if request.method == 'POST':
        target_col   = request.form.get('target')
        feature_cols = request.form.getlist('features')

        if not target_col or target_col not in columns:
            return "Invalid target selected"
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        if len(feature_cols) == 0:
            return "Please select at least one feature"

        session['target_col']   = target_col
        session['feature_cols'] = feature_cols

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            return render_template('ml_models_combined.html',
                                   columns=columns, selected_target=target_col,
                                   results=None, features=None,
                                   error_message="Selected target column must be numeric for regression.",
                                   filename=os.path.basename(filepath))

        results, features, cat_unique_values = train_supervised_models(df, target_col, feature_cols)
        session['cat_unique_values'] = cat_unique_values
        session['feature_cols']      = feature_cols

        return render_template('ml_models_combined.html',
                               columns=columns, selected_target=target_col,
                               results=results, features=features,
                               cat_unique_values=cat_unique_values,
                               original_feature_cols=feature_cols,
                               is_classification=False,
                               prediction=None, model_name=None, input_vector=None,
                               zip=zip, filename=os.path.basename(filepath))

    return render_template('ml_models_combined.html',
                           columns=columns, selected_target=None,
                           results=None, features=None,
                           cat_unique_values={}, original_feature_cols=[],
                           is_classification=None,
                           prediction=None, model_name=None, input_vector=None,
                           zip=zip, filename=os.path.basename(filepath))


@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    filepath   = session.get('filepath')
    target_col = session.get('target_col')

    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('index'))

    df         = pd.read_csv(filepath)
    model_name = request.form.get('model_name')

    if model_name not in trained_models:
        return "Model not found."

    model, trained_columns, scaler = trained_models[model_name]
    feature_cols       = session.get('feature_cols', [])
    cat_unique_values  = session.get('cat_unique_values', {})

    raw_input     = {}
    display_input = {}

    for col in feature_cols:
        val = request.form.get(col)
        if val is None:
            return f"Missing input for feature {col}"
        if col in cat_unique_values:
            raw_input[col]     = val
            display_input[col] = val
        else:
            raw_input[col]     = float(val)
            display_input[col] = float(val)

    raw_df   = pd.DataFrame([raw_input])
    raw_df   = pd.get_dummies(raw_df)
    input_df = raw_df.reindex(columns=trained_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]

    if not isinstance(prediction, (int, np.integer)):
        prediction = round(float(prediction), 2)

    results, features, cat_unique_values_fresh = train_supervised_models(df, target_col, feature_cols)

    return render_template("ml_models_combined.html",
                           filename=os.path.basename(filepath),
                           columns=list(df.columns),
                           results=results, features=features,
                           cat_unique_values=cat_unique_values_fresh,
                           original_feature_cols=feature_cols,
                           selected_target=target_col,
                           is_classification=False,
                           prediction=prediction,
                           model_name=model_name,
                           input_vector=[display_input[col] for col in feature_cols],
                           zip=zip)


if __name__ == '__main__':
    app.run(debug=True)