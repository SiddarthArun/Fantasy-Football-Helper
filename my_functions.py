import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math as m
from sklearn.linear_model import LinearRegression

#Data Loading Functions
def load_df_all():
    #Aggregate data from 2014-2024

    all_seasons = []

    for year in range(2014,2025):
        urls = [
        f"https://www.pro-football-reference.com/years/{year}/rushing.htm",
        f"https://www.pro-football-reference.com/years/{year}/receiving.htm"]

        #-----------------------------RUSHING DATA------------------------------------
        rudf = pd.read_html(urls[0])[0]

        # Flatten columns if multi-level
        if isinstance(rudf.columns, pd.MultiIndex):
            rudf.columns = [' '.join(col).strip() for col in rudf.columns.values]

        # Assign proper column names manually
        rudf.columns = [
            "Rk", "Player", "Age", "Team", "Pos", "G", "GS",
            "Rush_Att", "Rush_Yds", "Rush_TD", "Rush_1D", "Rush_Succ%",
            "Rush_Lng", "Rush_Y/A", "Rush_Y/G", "Rush_A/G", "Fmb", "Awards"
        ]

        # Drop columns you don't need
        df = rudf.drop(columns=["Rk", "Awards"])

        # Clean up column names
        df.columns = (
            df.columns.str.strip().str.replace(r'Unnamed.*', '', regex=True).str.replace('\n', ' ').str.strip())

        #-----------------------------RECEIVING DATA---------------------------------
        redf = pd.read_html(urls[1])[0]

        # Flatten columns if multi-level
        if isinstance(redf.columns, pd.MultiIndex):
            redf.columns = [' '.join(col).strip() for col in redf.columns.values]

        # Assign proper column names manually
        redf.columns = [
            "Rk", "Player", "Age", "Team", "Pos", "G", "GS",
            "Tgt", "Rec", "Yds", "Y/R", "TD",
            "1D", "Succ%", "Lng", "R/G", "Y/G", "Ctch%", "Y/Tgt", "Fmb", "Awards"
        ]

        # Drop columns you don't need
        redf = redf.drop(columns=["Rk", "Awards","Age", "Pos", "G", "GS", "Fmb"])

        # Clean up column names
        redf.columns = (
            redf.columns.str.strip().str.replace(r'Unnamed.*', '', regex=True).str.replace('\n', ' ').str.strip())

        #-------------------------------MERGE TABLES---------------------------------
        df = pd.merge(df, redf, on=['Player', 'Team'], how='left')
        df = df.drop(columns=['Rk'], errors ='ignore')

        #drop any NaNs and QBs
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        df = df[df["Player"] != "League Average"]


        #Add Fantasy Points  and Season columns
        df['Fpts'] = (
            df['Rush_Yds'] * 0.1 +
            df['Yds'] * 0.1 +
            df['Rush_TD'] * 6 + 
            df['TD'] * 6 + 
            df['Rec'] * 1
        )
        df["Fppg"] = df["Fpts"] / df['G'].replace(0,1)

        df['Season'] = year

        #Remove bums...
        df = df[df['Pos'].isin(['RB','WR','TE'])]
        df = df.sort_values('Fpts', ascending=False).head(125)
        df = df.sort_values('Fpts', ascending=False)

        #Add the df to the big list
        all_seasons.append(df)

    df_all = pd.concat(all_seasons, ignore_index=True)
    df_all.reset_index(drop=True, inplace=True)
    df_all = df_all.sort_values('Fpts', ascending=False)

    return df_all

def load_df_encoded(df_all):
    df_encoded = pd.get_dummies(df_all, columns=['Team', 'Pos'], dtype=int)
    return df_encoded


#Clustering
def run_clustering(df_encoded, df_all):
    #Isolate features to cluster by
    features_for_clustering = [
        'Age','Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A',
        'Rec', 'Yds', 'TD', 'Y/R', 'Y/Tgt',
        'Fppg', 'Fpts'
    ] + [col for col in df_encoded.columns if col.startswith('Pos_')]

    X = df_encoded[features_for_clustering]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Fit kmeans
    k=120
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_all['Cluster'] = kmeans.fit_predict(X_scaled)

    return df_all, features_for_clustering, kmeans

#Function to access season comparisons:
def get_comps(name, year, df_all):
    target = df_all[(df_all['Player'] == name) & (df_all['Season'] == year)]
    if target.empty:
        print('Player/Season not found.')
        return None

    cluster_id = target['Cluster'].iloc[0]
    comps = df_all[(df_all['Cluster'] == cluster_id) & ~((df_all['Player'] == name) & (df_all['Season'] == year))]
    return comps[['Player', 'Season', 'Fppg', 'Age']].sort_values('Fppg', ascending = False)

#Function for clustering plots

def plot_clusters(df_all, features_for_clustering, kmeans):
    fig, axes = plt.subplots(1,3, figsize = (18,6))

    #Graph 1: Age vs Fppg
    axes[0].scatter(df_all['Age'], df_all['Fppg'], c=df_all['Cluster'], cmap='tab20', alpha=0.6)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Fppg')
    axes[0].set_title('Age vs Fppg')

    #Graph 2: Rush_Attempts vs Fppg 
    axes[1].scatter(df_all['Rush_Att'], df_all['Fppg'], c=df_all['Cluster'], cmap='tab20', alpha=0.6)
    axes[1].set_xlabel("Rush Attempts")
    axes[1].set_ylabel("Fppg")
    axes[1].set_title("Rush_Att vs Fppg")

    #Graph 3: Feature Importance across clusters
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=features_for_clustering)

    sns.heatmap(centers_df, cmap='coolwarm', center=0, ax=axes[2])
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Clusters')
    axes[2].set_title('Cluster Centers (scaled values)')

    return fig



#Regression of Fpts
def train_regression_model(df_encoded):
    #sort by player and season, then shift FPPG up by a number of seasons
    lag_features = ['Fppg', 'Rush_Yds', 'Rec', 'TD', 'Tgt', 'Yds', 'Rush_TD', 'Rush_Att']
    n_lags = 4
    df_lag = df_encoded.sort_values(['Player','Season']).reset_index(drop=True)

    #Feature engineering relevant columns for regression
    df_lag['Age_squared'] = df_lag['Age']**2
    df_lag = df_lag.drop(columns=[col for col in df_lag.columns if col.startswith('Team_')])

    for lag in range(1, n_lags+1):
        for col in lag_features:
            df_lag[f'{col}_lag{lag}'] = df_lag.groupby('Player')[col].shift(lag)

    df_lag['Fppg_growth'] = (df_lag['Fppg_lag1'] - df_lag['Fppg_lag2']) / df_lag['Fppg_lag2']
    df_lag['Total_Touches'] = df_lag['Rec_lag1'] + df_lag['Rush_Att_lag1'] + df_lag['Tgt_lag1']
    df_lag['Pts_per_Touch'] = df_lag['Fppg_lag1'] / (df_lag['Total_Touches'] + 0.01)
    df_lag['Fppg_diff'] = df_lag['Fppg_lag1'] - df_lag['Fppg_lag2']
    df_lag['Target_Share'] = df_lag['Tgt_lag1'] / (df_lag['Tgt_lag1'] + df_lag['Rec_lag1'] + df_lag['Rush_Att_lag1'] + 0.01)
    df_lag['Fppg_pct_change'] = (df_lag['Fppg_lag1'] - df_lag['Fppg_lag2']) / df_lag['Fppg_lag2']
    df_lag['Fppg_next'] = df_lag.groupby('Player')['Fppg'].shift(-1)

    #make a copy of df_lag before drops for 2025 predictions
    df_features = df_lag.copy()

    df_lag = df_lag.dropna(subset=[f'{col}_lag{i}' for col in lag_features for i in range(1,n_lags+1)] + ['Fppg_next'])

    return df_lag, df_features

def train_model(df_lag):
    # Fitting and Training a Model
    X = df_lag.drop(columns=['Player','Fppg','Fppg_next','Fpts'])
    y = df_lag['Fppg_next']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = m.sqrt(mean_squared_error(y_test, y_pred))
    r2=r2_score(y_test,y_pred)

    return model, X_train,X_test, y_test, y_train, y_pred

def predict_player(player_name, season, feature_cols, model, df_features):
    player_row = df_features[(df_features['Player'] == player_name) & (df_features['Season'] == season-1)]

    if player_row.empty:
        return f'No Data found for {player_name} in {season}'
    
    X_player = player_row[feature_cols]

    prediction = model.predict(X_player)[0]

    return f'Predicted Fantasy points per game for {player_name} in {season} is {prediction:.2f}'

def plot_regression(X_train, y_test, y_pred, model):
    feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

    fig,axes = plt.subplots(1,2, figsize = (16,6))

    #Predicted vs expected
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0])
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel("Actual FPPG")
    axes[0].set_ylabel("Predicted FPPG")
    axes[0].set_title("Predicted vs Actual FPPG")

    #Top 15 feature importance
    sns.barplot(x='Importance', y='Feature', data = feat_imp.head(15), ax=axes[1])
    axes[1].set_title('Top 15 Most Important Features for Prediction')

    plt.tight_layout()
    plt.show()

    return fig

#Rankings
def prepare_shifted_df(df_all):
    id_cols = ['Player', 'Team', 'Pos', 'Season']

    stats_to_predict = ['Age', 'G', 'GS',
    'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_1D', 'Rush_Succ%', 'Rush_Lng', 'Rush_Y/A', 'Rush_Y/G', 'Rush_A/G', 'Fmb',
    'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'Lng', 'R/G', 'Y/G', 'Ctch%', 'Y/Tgt',
    'Fpts', 'Fppg']

    #make a copy and make a next year column for each stat
    df_shift = df_all.copy()
    for stat in stats_to_predict:
        df_shift[f'{stat}_next'] = df_shift.groupby('Player')[stat].shift(-1)

    #Get rid of any rows where there is no next year 
    df_shift = df_shift.dropna(subset=[f'{stat}_next' for stat in stats_to_predict])


    return df_shift, stats_to_predict

def predict_next_season(df_shift, stats_to_predict):
    feature_cols = [c for c in stats_to_predict if c not in ['Fpts', 'Fppg']]
    df_24 = df_shift[df_shift['Season'] == 2024].copy()
    df_preds = df_24.copy()

    for target in feature_cols:
        X = df_shift[feature_cols]
        y=df_shift[f'{target}_next']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        df_preds[f'{target}_proj'] = model.predict(df_24[feature_cols])

    return df_preds

def add_value_scores(df_preds):

    scaler = StandardScaler()
    df_preds_scaled = df_preds.copy()

    # Weighted Value Score
    df_preds_scaled["RB_ValueScore"] = (
        0.4*df_preds_scaled["Rush_Att_proj"] + 
        0.2*df_preds_scaled["Tgt_proj"] + 
        0.15*df_preds_scaled["Rush_Y/A_proj"] + 
        0.1*df_preds_scaled["Ctch%_proj"] +
        0.05*df_preds_scaled["Rush_Succ%_proj"] +
        0.05*df_preds_scaled["Succ%_proj"] +
        0.05*df_preds_scaled["Rush_1D_proj"]
    )

    df_preds_scaled["WRTE_ValueScore"] = (
        0.4*df_preds_scaled["Tgt_proj"] + 
        0.2*df_preds_scaled["Rec_proj"] +
        0.15*df_preds_scaled["Y/R_proj"] +
        0.1*df_preds_scaled["Ctch%_proj"] +
        0.05*df_preds_scaled["Succ%_proj"] +
        0.1*df_preds_scaled["1D_proj"]
    )

    df_preds['ValueScore'] = df_preds_scaled.apply(
        lambda row: row['RB_ValueScore'] if row['Pos'] == 'RB' else row['WRTE_ValueScore'], axis=1
    )

    return df_preds

def rank_players(df_preds):
    rb_df = df_preds[df_preds['Pos'] == 'RB'].copy()
    wr_df = df_preds[df_preds['Pos'] == 'WR'].copy()
    te_df = df_preds[df_preds['Pos'] == 'TE'].copy()

    rb_ranked = rb_df.sort_values(by='ValueScore', ascending=False).reset_index(drop=True)
    wr_ranked = wr_df.sort_values(by='ValueScore', ascending=False).reset_index(drop=True)
    te_ranked = te_df.sort_values(by='ValueScore', ascending=False).reset_index(drop=True)

    #add a rank column
    rb_ranked['PosRank'] = rb_ranked.index +1
    wr_ranked['PosRank'] = wr_ranked.index +1
    te_ranked['PosRank'] = te_ranked.index +1

    return rb_ranked, wr_ranked, te_ranked