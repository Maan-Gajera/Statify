import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("static/plots", exist_ok=True)

def get_columns(filepath):
    """Cleans data and categorizes columns for the UI"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip() 

    for col in df.columns:
        if df[col].dtype == "object":
            # Remove whitespace, $, and commas to fix "Amount"
            cleaned = df[col].astype(str).str.replace(r'[\$,\s]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            # If the column becomes numeric, update it
            if not converted.isna().all():
                df[col] = converted

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numeric_cols, cat_cols, df

def generate_plot(df, chart_type, x_axis=None, y_axis=None, z_axis=None):
    try:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Ensure Y-axis is numeric for charts that require it
        if y_axis and y_axis in df.columns and chart_type in ['bar', 'box', 'pie']:
             if not pd.api.types.is_numeric_dtype(df[y_axis]):
                df[y_axis] = pd.to_numeric(df[y_axis].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

        if chart_type == "histogram":
            sns.histplot(df[x_axis], kde=True, color="#4ecdc4")
            plt.title(f'Distribution of {x_axis}')

        elif chart_type == "bar":
            # Aggregates "Amount" by summing for "Country"
            sns.barplot(x=x_axis, y=y_axis, data=df, estimator=sum, palette="viridis")
            plt.title(f'Total {y_axis} by {x_axis}')
            plt.xticks(rotation=45)

        elif chart_type == "scatter":
            sns.scatterplot(x=x_axis, y=y_axis, data=df, s=100, color="#ff6b6b")
            plt.title(f'{x_axis} vs {y_axis}')

        elif chart_type == "box":
            sns.boxplot(x=x_axis, y=y_axis, data=df, palette="Set2")
            plt.title(f'{y_axis} Range by {x_axis}')

        elif chart_type == "heatmap":
            # Pivot requires Z to be numeric
            pivot = df.pivot_table(index=y_axis, columns=x_axis, values=z_axis, aggfunc="sum", fill_value=0)
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
            plt.title(f'Heatmap of {z_axis}')

        elif chart_type == "pie":
            data = df.groupby(x_axis)[y_axis].sum()
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
            plt.title(f'{y_axis} Share by {x_axis}')

        plot_filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join("static", "plots", plot_filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return plot_filename, []

    except Exception as e:
        print(f"Plot Error: {e}")
        plt.close()
        return None, []