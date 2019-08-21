# %% Import Libraries
from pandas import crosstab, DataFrame
from weather_gen import gen_weather_df

# %% Read Dataset
'''
Moved to a separate file for concise code.
'''
weather_df = gen_weather_df()

# %% View dataset
'''
Play is the dependent variable.
All other columns are independent variables.
'''
weather_df

# %% Overall outcome probabilities
'''
This will take the Play variable and divide
the total amount of Yes and No over the total records.

9/14 are Yes and 5/14 are No.
'''
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

# %% View proportions of Yes and No
'''
Dict includes the total + probability given the dataset.
The proportions are the initial probabilities.
'''
outcomes

# %% Generate frequency tables
'''
Now look at your independent variables and
how they occur with Yes and No outcomes.

This is essentially the setup to a chi2
analysis but we're not doing that here.
'''
freq_tables = {
    col: crosstab(
        weather_df[col],
        weather_df["Play"],
    )
    # Loop through all independent vars and
    # compare them to dependent var Play.
    for col in ('Outlook', 'Temperature', 'Humidity', 'Windy')
}

# %% Viewing frequencies all at once is harder to read
'''
Check the variable with the keys below to learn more
'''
freq_tables

# %% Weather conditions where No and Yes
freq_tables['Outlook']

# %% Temperature conditions where No and Yes
freq_tables["Temperature"]

# %% Humidity conditions where No and Yes
freq_tables['Humidity']

# %% Wind conditions where No and Yes
freq_tables['Windy']

# %% Update with conditional probabilities
'''
For each frequencey table, where Yes, divide by the
total amount of Yes overall.  Do the same for No.
'''
for table in freq_tables.values():
    table["P(No)"] = table["No"] / outcomes["No"]["Total"]
    table["P(Yes)"] = table["Yes"] / outcomes["Yes"]["Total"]
    table["Total"] = table["Yes"] + table["No"]
    table.loc["Total"] = table.sum()

# %% Weather conditions where No and Yes
'''
Each following table will now include a total
count summing to 14, the total amount of
overall records.

Essentially, keep dividing each cell by the total.
Example:

Where Play = No, there are 3 sunny days.  3/5 = 0.6.
Put that in cell Row=Sunny, Col=P(No).

Do this over and over.
'''
freq_tables['Outlook']

# %% Temperature conditions where No and Yes
freq_tables["Temperature"]

# %% Humidity conditions where No and Yes
freq_tables['Humidity']

# %% Wind conditions where No and Yes
freq_tables['Windy']

# %% Provide new record for a prediction
'''
Generally, new inputs in vector format, meaning
that they are a table of independent variables, are
a capital X.  Outputs are generally a single dependent
variable and commonly denoted with lower case y.

We're adding two new records that require prediction
of dependent variable y.
'''
y = {
    "Total":
    DataFrame({
        "Outlook": ("Sunny", "Overcast"),
        "Temperature": ("Cool", "Mild"),
        "Humidity": ("High", "Normal"),
        "Windy": ("True", "False"),
    })
}
X = y["Total"].copy()
y["Yes"] = y["Total"].copy()
y["No"] = y["Total"].copy()

# %% View input
X

# %% Yes, No, and Total are all initialized equal
y["Yes"]

# %% Yes, No, and Total are all initialized equal
y["No"]

# %% Yes, No, and Total are all initialized equal
y['Total']

# %% Initilize values for yes, no, total
'''
This is busy, but is taking all possible combinations
and again dividing instances over totals.
'''
for col in ('Outlook', 'Temperature', 'Humidity', 'Windy'):
    for val in y["Total"][col].unique():
        for key in ("Yes", "No", "Total"):
            y[key].loc[y[key][col] == val, col] = freq_tables[col][key][
                val] / freq_tables[col][key]["Total"]

# %% Adjust column names
keys = ("Outlook", "Temperature", "Humidity", "Windy")
y["Yes"].columns = [f"P({key}|Play=Yes)" for key in keys]
y["No"].columns = [f"P({key}|Play=No)" for key in keys]
y["Total"].columns = [f"P({key})" for key in keys]

# %% View Yes
'''
The '|' represents conditional probability.
Currently, the likelihood of independent
vars given that Play=Yes.
'''
y["Yes"]

# %% View No
'''
Likelihood of independent
vars given that Play=No.
'''
y["No"]

# %% View Totals
'''
The General proportions and probabilities
'''
y["Total"]

# %% Initialize result columns
'''
The X represents all probabilities being multiplied
together, Outlook * Temperature * Humidity * Windy
will be merged into X.  Use the outcomes
variable from the beginning.

Remember, probabilities should sum to 1 so set Total to that.
'''
y["Yes"]["P(X|Play=Yes)P(Play=Yes)"] = outcomes["Yes"]["P(Yes)"]
y["No"]["P(X|Play=No)P(Play=No)"] = outcomes["No"]["P(No)"]
y["Total"]["P(X)"] = 1

# %% View current yes
y["Yes"]

# %% View current no
y["No"]

# %% These should be the same as they're using prior data
'''
This could be considered the 'Machine Learning'
component.  You have prior probabilities being
merged into your current data.
'''
y["Yes"]["P(X|Play=Yes)P(Play=Yes)"]

# %% These should be the same as they're using prior data
y["No"]["P(X|Play=No)P(Play=No)"]

# %% Multiply all (P(X) should be 0.021865)
'''
We want the right most values.  Multiply
horizontally.
'''
y["Yes"]["P(X|Play=Yes)P(Play=Yes)"] = y["Yes"].prod(axis=1)
y["No"]["P(X|Play=No)P(Play=No)"] = y["No"].prod(axis=1)
y["Total"]["P(X)"] = y["Total"].prod(axis=1)

# %% Complete bayes theorum
y["Output"] = DataFrame({
    "P(Play=No|X)":
    y["No"]["P(X|Play=No)P(Play=No)"] / y["Total"]["P(X)"],
    "P(Play=Yes|X)":
    y["Yes"]["P(X|Play=Yes)P(Play=Yes)"] / y["Total"]["P(X)"],
})

# %% View the results
'''
They don't sum to 1 as there are two
parallel distinct processes for Yes and No
'''
y["Output"]

# %% Take the max for classification
y["Output"].loc[y["Output"]["P(Play=No|X)"] > y["Output"]["P(Play=Yes|X)"],
                "Play"] = "No"
y["Output"].loc[y["Output"]["P(Play=Yes|X)"] > y["Output"]["P(Play=No|X)"],
                "Play"] = "Yes"

# %% Assign back to X and view X
X["Play"] = y["Output"]["Play"]
X
