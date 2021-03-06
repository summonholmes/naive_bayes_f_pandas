# %% Import Libraries
from pandas import DataFrame


# %% Enter Data Here
def gen_weather_df():
    return DataFrame({
        "Outlook": (
            "Sunny",
            "Sunny",
            "Overcast",
            "Rain",
            "Rain",
            "Rain",
            "Overcast",
            "Sunny",
            "Sunny",
            "Rain",
            "Sunny",
            "Overcast",
            "Overcast",
            "Rain",
        ),
        "Temperature": (
            "Hot",
            "Hot",
            "Hot",
            "Mild",
            "Cool",
            "Cool",
            "Cool",
            "Mild",
            "Cool",
            "Mild",
            "Mild",
            "Mild",
            "Hot",
            "Mild",
        ),
        "Humidity": (
            "High",
            "High",
            "High",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "Normal",
            "Normal",
            "High",
            "Normal",
            "High",
        ),
        "Windy": (
            "False",
            "True",
            "False",
            "False",
            "False",
            "True",
            "True",
            "False",
            "False",
            "False",
            "True",
            "True",
            "False",
            "True",
        ),
        "Play": (
            "No",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "Yes",
            "No",
        ),
    })
