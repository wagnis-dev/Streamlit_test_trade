import streamlit as st
import pandas as pd
from pandas_datareader import data as pdr
import plotly.express as px
import numpy as np
from PIL import Image
import yfinance as yf



def parameter(df, sector_default_val, cap_default_val):
    
    # Secteur ##
    sector_values = [sector_default_val] + list(df['sector'].unique())
    option_sector = st.sidebar.selectbox("Sector", sector_values, index=0)
    # Fin secteur ##
    
    # Market capitalization ##
    cap_value_list = [cap_default_val] + ['Small', 'Medium', 'Large']
    cap_value = st.sidebar.selectbox("Capitalization", cap_value_list, index=0)
    # Fin Market capitalization
    
    # Dividend ##
    dividend_value = st.sidebar.slider('Dividend rate between than (%)', 0.0, 10.0, value=(0.0, 10.0))
    # Fin dividend ##
    
    # Profit ##
    
    min_profit_value, max_profit_value = float(df['profitMargins_%'].min()), float(df['profitMargins_%'].max())
    profit_value = st.sidebar.slider('Profit is margin greater than (%):', min_profit_value, max_profit_value, step = 10.0)
    # Fin profit ##
    
    
    
    return option_sector, cap_value, dividend_value,profit_value
    

# pour filtrer notre dataframe

def filterinf(df, sector_default_val, cap_default_val, option_sector, cap_value, dividend_value, profit_value):
    
    ## Sector filtering
    if option_sector != sector_default_val:
        df = df[df['sector'] == option_sector]
    
    
    ## Market capitalization filtering
    if cap_value != cap_default_val:
        if cap_value == 'Small':
            df = df[ (df['marketCap'] >= 0)
                 &
            (df['marketCap'] <= 20e9)]    
        elif cap_value == 'Medium':
            df = df[ (df['marketCap'] > 20e9)
                 &
            (df['marketCap'] <= 100e9)] 
        elif cap_value == 'Small':
            df = df[ df['marketCap'] > 100e9]
            
    ## Dividend
    df = df[
        (df['dividendYield_%'] >= dividend_value[0])
        &
        (df['dividendYield_%'] <= dividend_value[1])
       
    ]
    
    ## Profit
    
    df = df[df['profitMargins_%'] >= profit_value]
    
    
    return df        
               
   


# Creation de la methode d'import de données
@st.cache_data
def read_data():
    my_path = "udemy_streamlit/initial_version/project/s&p500.csv"
    df = pd.read_csv(my_path)
    return df


def company_price(df, option_company):
   
    # Vérification de l'entée
    if option_company is None or option_company not in df['name'].values:
        print("Entreprise non trouvée ou non sélectionnée.")
        return pd.DataFrame(columns = ['ds', 'y'])
    
    # Récupération du ticker
    ticker_company = df.loc[df['name'] == option_company, 'ticker'].values[0]
    
    try:
        # Téléchargement des données
        # Désactiver l'ajustement automatique pour conserver 'Adj Close'


        data_price = yf.download(ticker_company, start = '2014-12-31', end = '2024-12-31', auto_adjust=False)
        
        if 'Adj Close' not in data_price.columns:
            print("Colonne 'Adj Close' introuvable dans les données.")
            return pd.DataFrame(columns=['ds', 'y'])


        # Vérification des données
        if data_price.empty:
            print("Aucune données disponible pour ce tiker.")
            return pd.DataFrame(columns = ['ds', 'y'])
        
        # Préparation du DataFrame
        data_price = data_price[['Adj Close']].reset_index()
        data_price.columns = ['ds', 'y']
        return data_price
    
    except Exception as e:
        print(f"Erreur lors du téléchargement:{e}")
        return pd.DataFrame(columns = ['ds', 'y'])

def show_stock_price(price_data):
    fig = px.line(data_price, x = 'ds', y = 'y', title = '10 years stock price')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Stock price')
    st.plotly_chart(fig)


    
def metrics(data_price):
    stock_price_2015 = data_price['y'].values[0] #Take the first value (In most case it is 3 jan. 2012)
    stock_price_2025 = data_price['y'].values[-1] #Take the last value (In most case, it is 31 dec. 2021)
    performance = np.around((stock_price_2025/stock_price_2015 - 1)*100,2)
    return stock_price_2025,performance  
    

if __name__=="__main__":
    st.set_page_config(
        page_title = "Streamlit_PROJECT",# mettre un titre à l'affichage de ma page web
        page_icon="📈", # y mettre une icon
        initial_sidebar_state="expanded"   # lancer la sidebar au demarrage 
    )
    # mettre un titre à notre pas
    st.title("S&P500 Screener & Analysis")
    # mettre un titre à notre sidebar 
    st.sidebar.title("Search critera")
    # mettre une image dans notre page
    image = Image.open('udemy_streamlit/initial_version/project/stock.jpeg')
    # mettre une image au centre de la page en utilisant des colonnes
    _, col_image_2, _ = st.columns([1,3,1])
    with col_image_2:
        st.image(image, caption='ãustindistel')
    
    # Affichage du dataframe
    
    df = read_data()
    
    
    sector_default_val = 'All'
    cap_default_val = 'All'
    option_sector, cap_value, dividend_value,profit_value = parameter(df, sector_default_val, cap_default_val)
    df = filterinf(df, sector_default_val, cap_default_val, option_sector, cap_value, dividend_value, profit_value)
    
    
    st.subheader("Part 1 - S&P500 Screener")
    with st.expander("part 1 explanation", expanded = False):
        st.write(
            """
            Dans le tableau ci-dessous, vous trouverez la plupart des entreprises composant le S&P 500 (indice boursier regroupant les 500 plus grandes entreprises américaines), avec certains critères tels que :
                
                • 	Le nom de l’entreprise
                • 	Le secteur d’activité
                • 	La capitalisation boursière
                • 	Le pourcentage de distribution de dividendes (dividende/prix de l’action)
                • 	La marge bénéficiaire de l’entreprise en pourcentage
            
            ⚠️ Ces données sont extraites en temps réel depuis l’API Yahoo Finance. ⚠️
            
            ℹ️ Vous pouvez filtrer ou rechercher une entreprise à l’aide des filtres situés à gauche. ℹ️

        """
        )
    
    st.write('Number of compagnies found:', len(df))
    st.dataframe(df.iloc[:,:])
    
    # part 2 - Choose the company
    
    st.subheader("Part 2 - choose a company")
    option_company = st.selectbox("Choose a company : ", df.name.unique())
    
    # Part 3 - Stock analysis
    
    st.subheader("Part 3 - {} Stock Analysis".format(option_company))
    data_price = company_price(df, option_company)
    
    
    # Stock price analysis
    show_stock_price(data_price)
    
    stock_price_2025, performance = metrics(data_price)
    
    col_prediction_1,col_prediction_2 = st.columns([1,2])
    with col_prediction_1:
        st.metric(label="Stock price 31 dec. 2024", value=str(np.around(stock_price_2025,2)), delta=str(performance)+ ' %')
        st.write('*Compared to 31 dec. 2014*')

    with col_prediction_2:
        with st.expander("Prediction explanation",expanded=True):
            st.write("""
                Le graphique ci-dessus montre l’évolution du cours de l’action sélectionnée entre le 31 décembre 2014 et le 31 décembre 2024.
                L’indicateur situé à gauche affiche la valeur du cours de l’action au 31 décembre 2024 
                
                ⚠️pour l’entreprise sélectionnée, ainsi que son évolution entre le 31 décembre 2014 et le 31 décembre 2024. ⚠️ 
            """)