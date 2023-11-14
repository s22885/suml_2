# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

from unicodedata import numeric
import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model2.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model


ChestPainType = {1:'ATA',2:'NAP',0:'ASY',3:'TA'}
RestingECG = {1:'Normal',2:'ST',0:'LVH'}
ExerciseAngina = {1:'Y',0:'N'}
ST_Slope = {1:'Flat',2:'Up',0:'Down'}
sex_d = {0:"Female", 1:"Male"}

# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="serce s22885")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://sahyadrihospital.com/wp-content/uploads/2021/09/Heart-Attack-Symptoms.jpg")

	with overview:
		st.title("serce s22885")

	with left:
		Sexs = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		ChestPainTypes = st.radio( "ChestPainType", list(ChestPainType.keys()), format_func= lambda x: ChestPainType[x] )
		ExerciseAnginas = st.radio("ExerciseAngina",list(ExerciseAngina.keys()),format_func=lambda x: ExerciseAngina[x])
		ST_Slopes = st.radio("ST_Slope",list(ST_Slope.keys()),format_func=lambda x: ST_Slope[x])
		RestingECGs = st.radio("RestingECG",list(RestingECG.keys()),format_func=lambda x: RestingECG[x])

	with right:
		Ages = st.slider("Age", value=28, min_value=28, max_value=77)
		RestingBPs = st.slider("RestingBP", value=0, min_value=0, max_value=200)
		Cholesterols = st.slider("Cholesterol", value=0, min_value=0, max_value=603)
		FastingBSs = st.slider("FastingBS", value=0, min_value=0, max_value=1)
		MaxHRs = st.slider("MaxHR", value=60, min_value=60, max_value=202)
		Oldpeaks = st.slider("Oldpeak", value=-3, min_value=-3, max_value=6)

	data = [[Ages, Sexs, ChestPainTypes, RestingBPs, Cholesterols, FastingBSs, RestingECGs, MaxHRs, 
		ExerciseAnginas ,Oldpeaks ,ST_Slopes ]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy ma heart Disease?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
