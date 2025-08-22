import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from my_functions import load_df_all, load_df_encoded, run_clustering, plot_clusters, train_regression_model, train_model, plot_regression, prepare_shifted_df, get_comps, predict_player, predict_next_season, add_value_scores, rank_players


st.set_page_config(page_title="NFL Player Comps", layout="wide")
st.title('Fantasy Football Helper 🏈')
st.write('__________________________________________________________________________')

#Cache---------------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def _load_data():
    df_all=load_df_all()
    df_encoded = load_df_encoded(df_all)
    return df_all, df_encoded

@st.cache_data(show_spinner=True)
def _cluster_data():
    df_all, df_encoded = _load_data()
    df_clustered, features, kmeans = run_clustering(df_encoded, df_all.copy())
    return df_clustered, features, kmeans

@st.cache_data(show_spinner=True)
def _fpts_regression():
    df_all, df_encoded = _load_data()
    df_lag, df_features = train_regression_model(df_encoded)
    model, X_train, X_test, y_test, y_train, y_pred = train_model(df_lag)

    return df_features, model, X_train, X_test, y_test, y_pred

@st.cache_data(show_spinner=True)
def prep_rankings():
    df_all, df_encoded = _load_data()
    df_shift, stats_to_predict = prepare_shifted_df(df_all)
    df_preds = predict_next_season(df_shift, stats_to_predict)
    df_preds = add_value_scores(df_preds)
    rb_ranked, wr_ranked, te_ranked = rank_players(df_preds)

    return rb_ranked, wr_ranked, te_ranked

df_clustered, features, kmeans = _cluster_data()
df_features, model, X_train, X_test, y_test, y_pred = _fpts_regression()
rb_ranked, wr_ranked, te_ranked = prep_rankings()

#UI=============================================================================================

#Clustering
st.header('Find Season Comps 🔍')
colA, colB = st.columns(2)
with colA:
    name_input = st.text_input('Player Name: ', value='', placeholder='e.g. Saquon Barkley', key='cluster_inp')

    player_options = []
    if name_input:
        player_options = sorted(
            df_clustered.loc[
                df_clustered['Player'].str.contains(name_input, case=False, na=False), 'Player'
            ].unique()
        )
    player_selected = st.selectbox(
        'Select exact player', options=player_options if player_options else [], index=0 if player_options else None, placeholder='Start typing above', key='cluster_select'
    )

    season_selected = None
    if player_selected:
        seasons_available = sorted(df_clustered.loc[df_clustered['Player'] == player_selected, 'Season'].unique())

        season_selected = st.selectbox(
            'Season: ',
            options = seasons_available if len(seasons_available) else [],
            index = len(seasons_available)-1 if seasons_available else None
        )

    top_n = st.slider('How many comps to show',4,20,20,step=2)

    #Action
    go=st.button('Get Comps', disabled= not (player_selected and season_selected))

with colB:
    if go:
        comps = get_comps(player_selected, int(season_selected), df_clustered)

        if comps is None or comps.empty:
            st.warning('No comps found for this player/season')
        else:
            st.subheader(f'Closest comps to {season_selected} {player_selected}')
            st.dataframe(comps.head(top_n), use_container_width=True)

st.write('__________________________________________________________________________')
st.write('Insights on Clustering for Comps')
fig = plot_clusters(df_clustered, features, kmeans)
st.pyplot(fig)
st.write('__________________________________________________________________________')

#regression for fpts
st.header('Fantasy Points Predictor 📈')
cola, colb = st.columns([2,1])
with cola:
    name_input_r = st.text_input('Player Name: ', value='', placeholder='e.g. Saquon Barkley', key='fpts_pred_inp')

    player_options_r = []
    if name_input_r:
        player_options_r = sorted(
            df_features.loc[
                df_features['Player'].str.contains(name_input_r, case=False, na=False), 'Player'
            ].unique()
        )
    player_selected_r = st.selectbox(
        'Select exact player', options=player_options_r if player_options_r else [], index=0 if player_options_r else None, placeholder='Start typing above', key='fpts_pred_select'
    )

    go_r = st.button('Predict Fantasy Points', disabled=not player_selected_r)

with colb:
    if go_r:
        feature_cols = X_train.columns.tolist()
        prediction = predict_player(player_selected_r, 2025, feature_cols, model, df_features)
        st.header(prediction)
st.write('__________________________________________________________________________')
st.write('Insights for Regression on Fantasy Points per game')
fig_r = plot_regression(X_train, y_test, y_pred, model)
st.pyplot(fig_r)
st.write('__________________________________________________________________________')

#Rankings
st.header('Position Rankings 🥇')
col1, col2, col3 = st.columns(3)
with col1:
    st.write('Runningbacks')
    st.dataframe(rb_ranked[['PosRank','Player']].head(50))
with col2:
    st.write('Wide Recievers')
    st.dataframe(wr_ranked[['PosRank','Player', 'Fppg_next']].head(50))
with col3:
    st.write('Tight Ends')
    st.dataframe(te_ranked[['PosRank','Player', 'Fppg_next']].head(50))

#info
st.write('__________________________________________________________________________')
st.header('Some Info about the tool')
st.write('''
This tool uses Machine Learning on primarily historical data to make predictions about football. Team or coaching changes have the potential to completely throw off the stats presented here.
Additionally, the tool will not feature any rookies as they do not have any NFL data as of this release.

 - The comparisons presented with the comparison tool can be used to project how a player's season compares to other historical seasons and can be a good way of seeing the trajectory a
young player's career could take.
         
 - The Fantasy Points Prediction tool predicts fantasy output in 2025 based on historical trends and performance of a certain player.
         
 - The Rankings are based on predictions for future stats calculated using historic stats. It will prioritize players with more years of NFL experience and players who saw heavy usage in certain weeks,
so keep that in mind when some standout young players aren't on the lists and are replaced with linsanity runs It prioritizes short-term statistical upside over historic consistency.
         
For the comparison and fantasy points prediction tools, start typing a player's name in the "Player Name" field, and hit enter and use the "Select exact player field" if you are unsure of spelling.
''')