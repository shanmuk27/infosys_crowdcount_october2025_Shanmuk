# Using Matplotlib
import matplotlib.pyplot as plt
import pandas

x = [10, 20, 30, 40]
y = [20, 25, 35, 55]
plt.plot(x, y)
plt.title("Line Chart")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

a = [1, 2, 3, 4]
b = [10, 9, 8, 7]
plt.bar(a, b)
plt.title("Bar Chart")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

z = [6, 5, 4, 5, 4]
plt.hist(z, bins=5, color='steelblue', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

plt.scatter(x, y, color='red')
plt.title("Scatter Plot")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

values = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart")
plt.show()

l = [5, 4, 7, 8, 9]
plt.boxplot(l)
plt.title("Box Plot")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

plt.plot(x, y, color='green', linewidth=3, marker='o', markersize=10, linestyle='--')
plt.title("Customized Line Chart")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

# Using Seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.scatterplot(x='Name', y='Age', data=df)
plt.title("Seaborn Scatterplot")
plt.show()

sns.boxplot(y='Age', data=df)
plt.title("Seaborn Boxplot")
plt.show()

sns.violinplot(y='Age', data=df)
plt.title("Seaborn Violin Plot")
plt.show()

sns.swarmplot(x='Name', y='Age', data=df)
plt.title("Seaborn Swarm Plot")
plt.show()

sns.barplot(x='Name', y='Age', data=df)
plt.title("Seaborn Bar Plot")
plt.show()

sns.pointplot(x='Name', y='Age', data=df)
plt.title("Seaborn Point Plot")
plt.show()

sns.countplot(x='Name', data=df)
plt.title("Seaborn Count Plot")
plt.show()

# Using Plotly
import plotly.express as px

df = px.data.tips()

fig = px.line(df, x="day", y="total_bill", color='sex', title="Plotly Line Chart")
fig.show()

fig = px.bar(df, x='day', y="total_bill", color='sex', facet_col='time', title="Plotly Faceted Bar Chart")
fig.show()

fig = px.scatter(df, x='total_bill', y="tip", color='sex', title="Plotly Scatter Chart")
fig.show()

# Using Bokeh
from bokeh.plotting import figure, show

graph = figure(title="Bokeh Line Graph", x_axis_label='X', y_axis_label='Y')
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
graph.line(x, y, line_width=2, color='blue', legend_label='Line')
show(graph)
