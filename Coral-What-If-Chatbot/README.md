# Coral: Conversational What-If Process Analysis

This repository presents a conversational Chatbot to generate What-If scenarios in a business-related environment called Coral. Coral has the ability to read a BPMN model and make different changes in order to generate What-IF scenarios and help with business process optimization. This chatbot was built using the Python Framework for conversational chats called Rasa. In the same way, its interface was taken from a project available in GitHub repository that was built using React JS.

## Description 

What-If scenarios Chatbot is able to receive a text input and identify an intention in order to create a simulation scenario, then make the corresponding changes to the BPMN model, perform the simulation and return the respective results. On the other hand, What-If scenarios Chatbot is able to make comparisons between different scenarios that have been created using the chatbot and return the simulation results associated with those scenarios.

The conversational Chatbot allows specifying What-If scenarios based on the following changes in the process:

- Increase in process demand.
- Decrease in process demand.
- Addition of resources to the process.
- Modification of existing resources in the process.
- Removal of existing resources in the process.
- Optimization of process tasks (reduce task execution time).
- Delay of process tasks (increase task execution time).
- Creation of working timetables.
- Modification of working timetables.
- Process task automation.
- Compare generated models.

![Imagen](https://github.com/dfbaron/What-If-Chatbot/blob/main/images/Simulation%20Results.png)

## Input

The input for the What-If scenarios Chatbot is a BPMN model with the simulation parameters. From this model, the above-mentioned changes can be specified in the form of text inputs. Each scenario is associated with an intention in the chatbot, once a certain intention is predicted, some actions are executed, in which in some types of scenarios, additional information is requested to perform the modification of the BPMN model. 

## Output

Once all the scenario parameters are specified, the modifications are made and, the respective simulation of the process model with the embedded scenario is performed using BIMP Simulator. Once the simulation is done, the chatbot returns the simulation results of the base scenario (model without changes) and the simulation results of the created scenario.

## System Requirements

 - [Python 3.8](https://www.python.org/downloads/)
 - [Java SDK 1.8](https://www.oracle.com/fr/java/technologies/javase/javase8-archive-downloads.html)
 - [Anaconda Distribution](https://www.anaconda.com/products/individual)
 - [Git](https://git-scm.com/downloads)
 - [npm](https://nodejs.org/dist/v16.17.0/node-v16.17.0-x64.msi)

## Installation using conda

```python
conda create -n what_if_chatbot python=3.8
conda activate what_if_chatbot
pip install -U --user pip 
pip install rasa==3.2.1
git clone https://github.com/dfbaron/What-If-Chatbot.git
cd What-If-Chatbot
pip install -r requirements.txt
npm install react-scripts
```

## Use

Start anaconda prompt with what_if_chatbot environment and run:
```python
rasa run actions
```

Start another anaconda prompt with what_if_chatbot environment and run:
```python
rasa run --enable-api --cors "*"
```

Start a command line prompt and run:
```bash
cd RasaUI
npm start
```

## Example use

![ImagenConversacion](https://github.com/dfbaron/What-If-Chatbot/blob/main/images/Conversational%20Sample.png)
