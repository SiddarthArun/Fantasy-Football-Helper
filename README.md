# 🏈 Fantasy Football ML App
## Video demo at: https://youtu.be/kN5DVsNPC28

This project applies machine learning to a decade of NFL data (2014–2024) to provide insights and predictions for fantasy football. It combines clustering, regression, and statistical modeling to evaluate players, project performance, and rank skill positions.
Built with Python, scikit-learn, pandas, matplotlib, and Streamlit, the app enables users to explore data-driven fantasy insights interactively. IMPORTANT - This tool only covers skill positions: WR, RB, and TE. You will not find any data for QBs, Defenses, or Kickers here.

## 🚀 Features
🔎 Player Comparisons & Career Trajectories
- Uses KMeans clustering on individual player seasons to identify similar performance profiles.
- Helps uncover career arcs and compare players across seasons.

📈 Fantasy Points Prediction
- Random Forest Regression and Linear Regression trained on past fantasy performance.
- Predicts future fantasy points per game (PPG) for any player.

🏆 Positional Rankings
- Regression models on all past stats (usage, efficiency, value).
- Produces rankings for RB, WR, TE that:
- Prioritize short-term performance
- Reflect true player capability (not just raw fantasy totals)

📊 Data Visualizations
- Matplotlib charts to highlight feature importance and model insights.

## Tech Stack
- Python
- pandas - Data scraping, aggregation, and preprocessing
- scikit-learn - Clustering and regression models
- matplotlib/seaborn - Visualizations
- Streamlit - Web App interface
- Jupyter Notebook - Model development and experimentation

## Installation
- Download contents of this repository into a new folder
- Install python
- Run the following commands in the same directory as the project:
  
`pip install -r requirements.txt`

`streamlit run app.py`

## 📊 Example Outputs

- Clustered player seasons: See which players had comparable statistical profiles.
- Fantasy PPG predictions: Forecast future performance using regression models.
- Positional rankings: Evaluate which RB/WR/TEs provide the most value.
- Visual insights: Charts explaining what features the models prioritized (usage, efficiency, etc.).

## 🧠 Learning Highlights

This project demonstrates:
- End-to-end ML pipeline: preprocessing, training, evaluation, deployment.
- Combining unsupervised (KMeans) and supervised (regression) approaches.
- Building an interactive web app for non-technical users.
- Balancing short-term vs. long-term performance metrics in sports analytics.

## 🌟 Future Improvements

-  Incorporate more advanced models (XGBoost, Neural Nets).
- Add projections for rookie players using college-to-NFL transition data.
- Include injury and contract data as predictive features.
- Enhance UI with comparison dashboards and custom team-building tools.

## Why Fantasy Football?

- Massive Popularity - Played by millions worldwide
- Data rich environment - every snap and yard is logged and provides extremely high quality data to work with
- Predictive challenge - creating an effective tool requires smart use of stats and a good understanding of the sport
- Decision making under uncertainty - Fantasy football simulates real-world data science problems like portfolio management, risk/reward tradeofs, and optimization
- Very fun to work on...
