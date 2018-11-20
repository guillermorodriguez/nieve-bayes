"""
    @ Author:       Guillermo Rodriguez
    @ Date:         09/18/2018
    @ Purpose:      Creates a bar chart of a data series with labels
    @ Dependency:   Plotly.ly
                    pip install plotly
"""

import plotly
import os

class graph():

    """
        Constructor
    """
    def __init__(self):
        print("Graph Object Initialized")
        print("Plotly Version: %s" % plotly.__version__)

    """
        @ Author:       Guillermo Rodriguez
        @ Date:         09/18/2018
        @ Purpose:      Creates a bar chart from input sequence
    """
    def create_plot(self, input):
        labels = []
        points = []
        path = os.getcwd() + '\\Charts\\nieve-bayes.html'

        for key in input.keys():
            labels.append(key)
            points.append(input[key])

        data = [
           plotly.graph_objs.Bar(
                x=labels,
                y=points
            )
        ]

        plotly.offline.plot(data, filename=path)
