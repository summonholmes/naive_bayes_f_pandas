# %% Import Libraries
from pandas import crosstab, DataFrame, read_pickle

# %% Read Dataset
weather_df = read_pickle("weather_df.pkl")

# %% Overall outcome probabilities
outcomes = {
    outcome: {
        "Total":
        weather_df.loc[weather_df["Play"] == outcome].shape[0],
        f"P({outcome})":
        weather_df.loc[weather_df["Play"] == outcome].shape[0] /
        weather_df.shape[0],
    }
    for outcome in ("Yes", "No")
}

# %% Generate frequency tables
freq_tables = {
    col: crosstab(
        weather_df[col],
        weather_df["Play"],
    )
    for col in weather_df.columns.drop("Play")
}

# %% Update with conditional probabilities
for table in freq_tables.values():
    table["P(Yes)"] = table["Yes"] / outcomes["Yes"]["Total"]
    table["P(No)"] = table["No"] / outcomes["No"]["Total"]
    table["Total"] = table["Yes"] + table["No"]
    table.loc["Total"] = table.sum()

# %% Provide Dataset
X_1 = {
    "Total":
    DataFrame({
        "Outlook": ("Sunny", "Overcast"),
        "Temperature": ("Cool", "Mild"),
        "Humidity": ("High", "Normal"),
        "Windy": ("True", "False"),
    })
}
X_1["Yes"] = X_1["Total"].copy()
X_1["No"] = X_1["Total"].copy()

# %% Initilize values for yes, no, total
keys = ("Yes", "No", "Total")
for col in X_1["Total"].columns:
    for val in X_1["Total"][col].unique():
        for key in keys:
            X_1[key].loc[X_1[key][col] == val, col] = freq_tables[col][key][
                val] / freq_tables[col][key]["Total"]

# %% Adjust column names
keys = ("Outlook", "Temperature", "Humidity", "Windy")
X_1["Yes"].columns = [f"P({key}|Play=Yes)" for key in keys]
X_1["No"].columns = [f"P({key}|Play=No)" for key in keys]
X_1["Total"].columns = [f"P({key})" for key in keys]

# %% Initialize result columns
X_1["Yes"]["P(X|Play=Yes)P(Play=Yes)"] = outcomes["Yes"]["P(Yes)"]
X_1["No"]["P(X|Play=No)P(Play=No)"] = outcomes["No"]["P(No)"]
X_1["Total"]["P(X)"] = 1

# %% Multiply all (P(X) should be 0.021865)
X_1["Yes"]["P(X|Play=Yes)P(Play=Yes)"] = X_1["Yes"].prod(axis=1)
X_1["No"]["P(X|Play=No)P(Play=No)"] = X_1["No"].prod(axis=1)
X_1["Total"]["P(X)"] = X_1["Total"].prod(axis=1)

# %% Get probabilities for yes
X_1["Yes"]["P(X|Play=Yes)P(Play=Yes)"][0] / X_1["Total"]["P(X)"][0]

# %% Get probabilities for no
X_1["No"]["P(X|Play=No)P(Play=No)"][0] / X_1["Total"]["P(X)"][0]
