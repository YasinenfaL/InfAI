"""
DataAnalyzerApp: A Python application for data analysis, correlation analysis,
and conversational interaction using Streamlit and the LangChain library.
"""

import os
import sys
import glob
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("new_project")
from data_utils import summarize_csv_with_model, analyze_trend, ask_question


class DataAnalyzerApp:
    """
    A Streamlit-based application for data analysis and interactive chat.
    """

    def __init__(self):
        """
        Initializes the application, loads environment variables, sets up the language model, and prepares Streamlit components.
        """
        self.openai_api_key = None
        self.image_save_path = None
        self.csv_directory = None
        self.llm_gpt = None
        self.system_message = None
        self.selection = None
        self.dataframes = {}  # Birden fazla veri çerçevesini tutmak için

        self.load_environment()
        self.setup_llm()
        self.load_data()
        self.setup_streamlit()

    def load_environment(self):
        """
        Loads environment variables from the .env file and sets the API key, image save path, and CSV directory.
        """
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.image_save_path = os.getenv("IMAGE_SAVE_PATH", "images")
        self.csv_directory = os.getenv("CSV_DIRECTORY", "data")

        if not self.openai_api_key:
            st.error("OpenAI API anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")
            st.stop()

    def setup_llm(self):
        """
        Sets up the language model using the OpenAI API.
        """
        self.llm_gpt = ChatOpenAI(
            api_key=self.openai_api_key,
            model="gpt-4",
            temperature=0.2,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.system_message = SystemMessage(
            content=(

            )
        )

    def load_data(self):
        """
        Loads CSV files from the specified directory and assigns them to dataframes.
        """

        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))

        if not csv_files:
            st.error(f"{self.csv_directory} dizininde CSV dosyası bulunamadı.")
            st.stop()

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            file_name = os.path.basename(csv_file)
            self.dataframes[file_name] = df

    def setup_streamlit(self):
        """
        Prepares the Streamlit interface, title, and sidebar menu.
        """
        st.set_option('client.showErrorDetails', True)
        st.title("Inf AI 🤖")
        self.sidebar_menu()
        self.main()

    def sidebar_menu(self):
        """
        Defines the sidebar menu for navigation in the Streamlit interface.
        """
        st.sidebar.title("Menü")
        self.selection = st.sidebar.radio(
            "Lütfen bir seçenek seçin:",
            ["Veri Analizi", "Korelasyon Analizi", "Sohbet Et"]
        )

    def main(self):
        """
        Directs the user to different modules based on the selection in the sidebar.
        Processes all CSV files.
        """
        if self.selection == "Veri Analizi":
            self.data_analysis()
        elif self.selection == "Korelasyon Analizi":
            self.trend_analysis()
        elif self.selection == "Sohbet Et":
            self.question_answering()

    def data_analysis(self):
        """
        Performs data analysis for each CSV file.
        """
        for file_name, df in self.dataframes.items():
            st.header(f"{file_name} - Veri Seti Özeti")
            st.subheader("Inf AI 🤖 ile Analiz")
            summary = summarize_csv_with_model(df, self.llm_gpt)

            st.write("**İlk 5 Satır**")
            st.write(df.head())

            st.write("**Kolon Açıklamaları**")
            st.write(summary["column_descriptions"])

            st.write("**Eksik Veriler**")
            st.write(summary["missing_values"])

            st.write("**Çift Kayıtlar**")
            st.write(summary["duplicate_values"])

            st.write("**Anormal Değerler**")
            st.write(summary["anomaly_values"])

            st.write("**Temel İstatistikler**")
            st.write(summary["essential_metrics"])

            st.write("**Veri Tipleri**")
            st.write(summary["data_types"])

            st.write("**Veri Seti Boyutu**")
            st.write(df.shape)

            os.makedirs(self.image_save_path, exist_ok=True)

            st.header("Sayısal Kolonların Dağılımı")
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for column in numeric_columns:
                st.subheader(f'{column} Dağılımı')
                plt.figure(figsize=(10, 4))
                sns.histplot(df[column], kde=True)
                plt.tight_layout()

                image_path = os.path.join(self.image_save_path, f'{file_name}_{column}_distribution.png')
                plt.savefig(image_path)
                st.image(image_path)
                plt.clf()

    def trend_analysis(self):
        """
        Performs correlation analysis for each CSV file.
        """
        variable_of_interest = st.text_input("Analiz Etmek İstediğiniz Değişkeni Girin")

        if variable_of_interest:
            for file_name, df in self.dataframes.items():
                st.header(f"{file_name} - Korelasyon Analizi")
                trend_response = analyze_trend(df, self.llm_gpt, variable_of_interest)
                st.write(trend_response)

    def question_answering(self):
        """
        Answers user questions for each CSV file.
        """
        user_question = st.text_input("Sorularınızı Buraya Yazın")

        if user_question:
            for file_name, df in self.dataframes.items():
                st.header(f"{file_name} - Sohbet Et")
                st.subheader(f"Soru: {user_question}")
                ai_response = ask_question(df, self.llm_gpt, user_question)
                st.write(ai_response)


if __name__ == "__main__":
    app = DataAnalyzerApp()
