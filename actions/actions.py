# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from asyncore import dispatcher
from dis import dis
from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_core_sdk.forms import FormAction
from rasa_sdk.events import SlotSet
from rasa_sdk.events import AllSlotsReset
from typing import Dict, Text, List
from rasa_sdk.events import EventType


import numpy as np
import pandas as pd
import re
from glob import glob
import string
import random
import uuid
from datetime import datetime
import time
import json
import os

from . import utils_chatbot as u

class ValidateAddResourcesForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_add_resources_form"

    @staticmethod
    def add_resource_time_table_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_timetables = u.extract_timetables(model_path)

        return list(df_timetables['timetableName'])

    @staticmethod
    def add_resource_new_role_db(tracker) -> List[Text]:
        """Database of supported roles for new role."""

        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)

        return list(df_tasks['taskName'])

    @staticmethod
    def add_resource_name_db(tracker) -> List[Text]:
        """Database of supported roles for new role."""

        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)

        return list(df_resources['resourceName'])

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""

        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate_add_resource_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_name value."""

        resources = self.add_resource_name_db(tracker)

        if value.lower() not in [x.lower() for x in resources]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"add_resource_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_name")
            for resource in resources:
                dispatcher.utter_message(resource)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"add_resource_name": None}

    def validate_add_resource_time_table(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_time_table value."""

        timetables = self.add_resource_time_table_db(tracker)

        if value.lower() in [x.lower() for x in timetables]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"add_resource_time_table": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_time_table")
            for timetable in timetables:
                dispatcher.utter_message(timetable)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"add_resource_time_table": None}

    def validate_add_resource_new_role(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_new_role value."""

        tasks = self.add_resource_new_role_db(tracker)
        
        if value.lower() in [x.lower() for x in tasks]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"add_resource_new_role": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_new_role")
            for task in tasks:
                dispatcher.utter_message(task)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"add_resource_new_role": None}

    def validate_add_resource_amount(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_amount value."""

        if self.is_int(value) and int(value) > 0:
            return {"add_resource_amount": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_amount")
            # validation failed, set slot to None
            return {"add_resource_amount": None}

    def validate_add_resource_cost(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_cost value."""

        if self.is_int(value) and int(value) > 0:
            return {"add_resource_cost": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_cost")
            # validation failed, set slot to None
            return {"add_resource_cost": None}

class AddResourcesForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "add_resources_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["add_resource_name", "add_resource_amount", "add_resource_cost",
                "add_resource_time_table", "add_resource_new_role"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """
        return []

class ActionHelp(Action):
    def name(self) -> Text:
        return "action_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        msg = """   Coral support these what-if scenarios:
    - Increase of demand
    - Decrease of demand
    - Adding resources
    - Modifying resources
    - Removing resources
    - Optimizing tasks
    - Creating new timetables
    - Modifying timetables
    - Automating tasks
    - Compare generated models"""

        msgs = msg.split('\n')
        for msg_h in msgs:
            dispatcher.utter_message(text=msg_h)

class AskForAddResourceTimeTable(Action):
    def name(self) -> Text:
        return "action_ask_add_resource_time_table"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which of these timetables?")
        
        model_path = tracker.get_slot("model")
        df_timetables = u.extract_timetables(model_path)
        for timetable in df_timetables['timetableName']:
            dispatcher.utter_message(text=timetable)
        
        return []

class AskForAddResourceNewRole(Action):
    def name(self) -> Text:
        return "action_ask_add_resource_new_role"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="To which task do you want to assign the new resource?")
        
        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)
        for task in df_tasks['taskName']:
            dispatcher.utter_message(text=task)
        
        return []
  
class ActionAddResource(Action):
    def name(self) -> Text:
        return "action_add_resources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)
        df_timetables = u.extract_timetables(model_path)
        
        resourceId = 'qbp_{}'.format(uuid.uuid4())
        resourceName = tracker.get_slot("add_resource_name")
        totalAmount = tracker.get_slot("add_resource_amount")
        costPerHour = tracker.get_slot("add_resource_cost")

        timetableName = tracker.get_slot("add_resource_time_table")
                
        timetableId = df_timetables[df_timetables['timetableName']== timetableName]['timetableId'].values[0]
        
        df_new_role = pd.DataFrame([{'resourceId':resourceId, 'resourceName':resourceName, 'totalAmount':totalAmount, \
                            'costPerHour':costPerHour, 'timetableId':timetableId}])
        df_resources = pd.concat([df_resources, df_new_role])
        
        df_elements = u.extract_elements(model_path)
        df_tasks = u.extract_tasks(model_path)
        df_tasks_elements = df_tasks.merge(df_elements, how='left', on='elementId')
        df = df_tasks_elements[['taskName', 'elementId', 'resourceId']].merge(df_resources, how='left', on='resourceId')
        
        df_transformed = df.copy()

        task_new_role = tracker.get_slot("add_resource_new_role")
        df_transformed.loc[df_transformed['taskName'].str.lower() == task_new_role.lower(), 'resourceId'] = resourceId
        df_transformed.loc[df_transformed['taskName'].str.lower() == task_new_role.lower(), 'resourceName'] = resourceName
        df_transformed.loc[df_transformed['taskName'].str.lower() == task_new_role.lower(), 'totalAmount'] = totalAmount
        df_transformed.loc[df_transformed['taskName'].str.lower() == task_new_role.lower(), 'costPerHour'] = costPerHour
        df_transformed.loc[df_transformed['taskName'].str.lower() == task_new_role.lower(), 'timetableId'] = timetableId

        ptt_s = '<qbp:elements>'
        ptt_e = '</qbp:elements>'
        elements = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)
        element_lines = elements.split('\n')
        elements_list = []
        start, end = None, None
        for idx, line in enumerate(element_lines):
            if '<qbp:element ' in line and start == None:
                start = idx
            if '</qbp:element>' in line and end == None:
                end = idx
            if start != None and end != None:
                elements_list.append('\n'.join(element_lines[start:end+1]))
                start, end = None, None
                
        df = df.sort_values(by='taskName')
        df_transformed = df_transformed.sort_values(by='taskName')
        
        # Extract new elements and replace old one with new elements extracted
        new_elements = []
        for i in range(len(elements_list)):
            element = elements_list[i]
            old_elem = list(df[df['taskName'].str.lower() == task_new_role.lower()]['resourceId'])[0]
            new_elem = list(df_transformed[df_transformed['taskName'].str.lower() == task_new_role.lower()]['resourceId'])[0]
            if 'elementId="{}"'.format(list(df[df['taskName'].str.lower() == task_new_role.lower()]['elementId'])[0]) in element:
                new_element = element.replace(old_elem, new_elem)
                new_elements.append(new_element)
        
        new_elements = '\n'.join([element_lines[0]] + new_elements + [element_lines[-1]])
        
        with open(model_path) as file:
            model= file.read()
        new_model = model.replace(elements, new_elements)        
        
        ptt_s = '<qbp:resources>'
        ptt_e = '</qbp:resources>'
        resources = u.extract_bpmn_resources(model_path, ptt_s, ptt_e).split('\n')
        new_res = '      <qbp:resource id="{}" name="{}" totalAmount="{}" costPerHour="{}" timetableId="{}"/>'.format(resourceId, resourceName, totalAmount, \
                                                                                                                costPerHour, timetableId)
        new_resources = '\n'.join(resources[:-1] + [new_res] + [resources[-1]])
        new_model = new_model.replace('\n'.join(resources), new_resources)
        
        new_model_path = model_path.split('.')[0] + '_add_resource_{}'.format(resourceName.replace(' ', '_')) + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/resources/models')
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
            
        sce_name = resourceName.replace(' ', '_')
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_add_resource_{}.csv'.format(sce_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Resource Addition')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the Baseline Scenario')
            
        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)
        
        return [SlotSet("add_resource_name", None),
                SlotSet("add_resource_amount", None),
                SlotSet("add_resource_cost", None),
                SlotSet("add_resource_time_table", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("add_resource_new_role", None)]

class ActionIncreaseDemand(Action):
    """
    Action for increase demand
    """

    def name(self) -> Text:
        return "action_increase_demand"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")


        input_m = next(tracker.get_latest_entity_values("inc_percentage"))
        inc_percentage = float(input_m)
        percentage = inc_percentage/100 if inc_percentage > 1 else inc_percentage 
        sce_name = str(int(percentage*100))
        new_model_path = u.modify_bimp_model_instances(model_path, percentage)
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/demand/output_inc_demand_{}.csv'.format(sce_name)
       
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)

        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Increase Demand')

        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/demand/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the Baseline Scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(output_message)

        return [SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name)]

class ActionDecreaseDemand(Action):
    """
    Action for decrease demand
    """

    def name(self) -> Text:
        return "action_decrease_demand"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        input_m = next(tracker.get_latest_entity_values("dec_percentage"))
        dec_percentage = float(input_m)
        percentage = dec_percentage/100 if np.abs(dec_percentage) > 1 else dec_percentage
        p = int(np.abs(percentage)*100)
        sce_name = 'Decreased demand in {} percent'.format(p)

        new_model_path = u.modify_bimp_model_instances(model_path, percentage)

        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/demand/output_dec_demand_{}.csv'.format(sce_name)
        
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)

        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Decrease demand')

        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/demand/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the Baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(output_message)
        
        return [SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name)]

class ValidateChangeResourcesForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_change_resources_form"

    @staticmethod
    def change_resources_role_modify_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)

        return list(df_resources['resourceName'])

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""

        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate_change_resources_role_modify(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_name value."""

        resources = self.change_resources_role_modify_db(tracker)

        if value.lower() in [x.lower() for x in resources]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"change_resources_role_modify": value}
        else:
            dispatcher.utter_message(response="utter_wrong_change_resources_role_modify")
            for resource in resources:
                dispatcher.utter_message(resource)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"change_resources_role_modify": None}

    def validate_change_resources_new_amount(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate change_resources_new_amount value."""

        if self.is_int(value) and int(value) > 0:
            return {"change_resources_new_amount": value}
        else:
            dispatcher.utter_message(response="utter_wrong_add_resource_amount")
            # validation failed, set slot to None
            return {"change_resources_new_amount": None}

    def validate_change_resources_new_cost(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate change_resources_new_cost value."""

        if self.is_int(value) and int(value) > 0:
            return {"change_resources_new_cost": value}
        else:
            dispatcher.utter_message(response="utter_wrong_change_resources_new_cost")
            # validation failed, set slot to None
            return {"change_resources_new_cost": None}

class ChangeResourcesForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "change_resources_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["change_resources_role_modify", "change_resources_new_amount", "change_resources_new_cost"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class AskForChangeResourceRoleModify(Action):
    def name(self) -> Text:
        return "action_ask_change_resources_role_modify"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which resource do you want to modify?")
        
        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)
        for resource in df_resources['resourceName']:
            dispatcher.utter_message(text=resource)
        
        return []

class ActionChangeResource(Action):
    def name(self) -> Text:
        return "action_change_resources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        df_resources = u.extract_resources(model_path)
        
        mod_res = tracker.get_slot("change_resources_role_modify")
        
        new_amount = tracker.get_slot("change_resources_new_amount")
        new_cost = tracker.get_slot("change_resources_new_cost")
        
        df_resources.loc[df_resources['resourceName']==mod_res, ['totalAmount']] = new_amount
        df_resources.loc[df_resources['resourceName']==mod_res, ['costPerHour']] = new_cost
        
        mod_name = mod_res.replace(' ', '_')

        resources = """    <qbp:resources>
            {} 
        </qbp:resources>"""

        resource = """<qbp:resource id="{}" name="{}" totalAmount="{}" costPerHour="{}" timetableId="{}"/>"""
        df_resources['resource'] = df_resources.apply(lambda x: resource.format(x['resourceId'], x['resourceName'], x['totalAmount'], \
                                                                                x['costPerHour'], x['timetableId']
                                                                                ), axis=1)
        new_resources = resources.format("""""".join(df_resources['resource']))

        with open(model_path) as f:
            model = f.read()

        ptt_s = '<qbp:resources>'
        ptt_e = '</qbp:resources>'
        resources_text = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)

        new_model = model.replace(resources_text, new_resources)

        sce_name = '_mod_resource_{}'.format(mod_name)
        new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/resources/models')

        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
        
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_mod_resource_{}.csv'.format(mod_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Resource Modification')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)

        return [SlotSet("change_resources_role_modify", None),
                SlotSet("change_resources_new_amount", None),
                SlotSet("change_resources_new_cost", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", mod_name)]

class ValidateFastTaskForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_fast_task_form"

    @staticmethod
    def fast_task_name_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_tasks, task_dist = u.extract_task_add_info(model_path)

        return list(df_tasks['name'])

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""

        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate_fast_task_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_name value."""

        tasks = self.fast_task_name_db(tracker)

        if value.lower() in [x.lower() for x in tasks]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"fast_task_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_fast_task_name")
            for task in tasks:
                dispatcher.utter_message(task)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"fast_task_name": None}

    def validate_fast_task_percentage(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate fast_task_percentage value."""

        if self.is_int(value) and int(value) > 0 and int(value) < 100:
            return {"fast_task_percentage": value}
        else:
            dispatcher.utter_message(response="utter_wrong_fast_task_percentage")
            # validation failed, set slot to None
            return {"fast_task_percentage": None}

class AskForFastTaskName(Action):
    def name(self) -> Text:
        return "action_ask_fast_task_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which task do you want to become faster?")
        
        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)
        for task in df_tasks['taskName']:
            dispatcher.utter_message(text=task)
        
        return []

class FastTaskForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "fast_task_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["fast_task_name", "fast_task_percentage"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class ActionFastTask(Action):
    def name(self) -> Text:
        return "action_fast_task"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        df_tasks, task_dist = u.extract_task_add_info(model_path)
        
        task = tracker.get_slot("fast_task_name")

        percentage = int(tracker.get_slot("fast_task_percentage"))/100

        df_tasks.loc[df_tasks['name'].str.lower() == task.lower(), ['mean']] = (1-percentage)*int(df_tasks[df_tasks['name'].str.lower() == task.lower()]['mean'].values[0])

        elements = """
            <qbp:elements>
                {}
            </qbp:elements>
        """
        
        element = """      <qbp:element id="{}" elementId="{}">
                <qbp:durationDistribution type="{}" mean="{}" arg1="{}" arg2="{}">
                <qbp:timeUnit>{}</qbp:timeUnit>
                </qbp:durationDistribution>
                <qbp:resourceIds>
                <qbp:resourceId>{}</qbp:resourceId>
                </qbp:resourceIds>
            </qbp:element>
        """
        
        df_tasks['element'] = df_tasks.apply(lambda x: element.format(x['id'], x['elementId'], x['type'], x['mean'], \
                                                                    x['arg1'], x['arg2'], x['timeUnit'], x['resourceId']), \
                                             axis= 1)
            
        new_elements = elements.format("""""".join(df_tasks['element']))
        
        with open(model_path) as file:
            model= file.read()

        new_model = model.replace('\n'.join(task_dist[0]), new_elements)
        sce_name = '_{}_faster_{}'.format(percentage, task)
        
        new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/fast_slow_task/models')
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
            
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/fast_slow_task/output_{}.csv'.format(sce_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Faster Task')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/fast_slow_task/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)

        return [SlotSet("fast_task_name", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("fast_task_percentage", None)]

class ValidateSlowTaskForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_slow_task_form"

    @staticmethod
    def slow_task_name_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_tasks, _ = u.extract_task_add_info(model_path)

        return list(df_tasks['name'])

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""

        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate_slow_task_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate add_resource_name value."""

        tasks = self.slow_task_name_db(tracker)

        if value.lower() in [x.lower() for x in tasks]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"slow_task_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_slow_task_name")
            for task in tasks:
                dispatcher.utter_message(task)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"slow_task_name": None}

    def validate_slow_task_percentage(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate slow_task_percentage value."""

        if self.is_int(value) and int(value) > 0 and int(value) < 100:
            return {"slow_task_percentage": value}
        else:
            dispatcher.utter_message(response="utter_wrong_slow_task_percentage")
            # validation failed, set slot to None
            return {"slow_task_percentage": None}

class AskForSlowTaskName(Action):
    def name(self) -> Text:
        return "action_ask_slow_task_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which task do you want to become slower?")
        
        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)
        for task in df_tasks['taskName']:
            dispatcher.utter_message(text=task)
        
        return []

class SlowTaskForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "slow_task_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["slow_task_name", "slow_task_percentage"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class ActionSlowTask(Action):
    def name(self) -> Text:
        return "action_slow_task"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        df_tasks, task_dist = u.extract_task_add_info(model_path)
        
        task = tracker.get_slot("slow_task_name")
        percentage = int(tracker.get_slot("slow_task_percentage"))/100

        df_tasks.loc[df_tasks['name'].str.lower() == task.lower(), ['mean']] = (1+percentage)*int(df_tasks[df_tasks['name'].str.lower() == task.lower()]['mean'].values[0])

        elements = """
            <qbp:elements>
                {}
            </qbp:elements>
        """
        
        element = """      <qbp:element id="{}" elementId="{}">
                <qbp:durationDistribution type="{}" mean="{}" arg1="{}" arg2="{}">
                <qbp:timeUnit>{}</qbp:timeUnit>
                </qbp:durationDistribution>
                <qbp:resourceIds>
                <qbp:resourceId>{}</qbp:resourceId>
                </qbp:resourceIds>
            </qbp:element>
        """
        
        df_tasks['element'] = df_tasks.apply(lambda x: element.format(x['id'], x['elementId'], x['type'], x['mean'], \
                                                                    x['arg1'], x['arg2'], x['timeUnit'], x['resourceId']), \
                                            axis= 1)
            
        new_elements = elements.format("""""".join(df_tasks['element']))
        
        with open(model_path) as file:
            model= file.read()

        new_model = model.replace('\n'.join(task_dist[0]), new_elements)
        sce_name = '_{}_slower_{}'.format(percentage, task)
        
        new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/fast_slow_task/models')
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
            
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/fast_slow_task/output_{}.csv'.format(sce_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Slower Task')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/fast_slow_task/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)

        return [SlotSet("slow_task_name", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("slow_task_percentage", None)]

class ValidateRemoveResourceskForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_remove_resources_form"

    @staticmethod
    def remove_resources_role_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)

        return list(df_resources['resourceName'])

    def validate_remove_resources_role(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate remove_resources_role value."""

        resources = self.remove_resources_role_db(tracker)

        if value.lower() in [x.lower() for x in resources]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"remove_resources_role": value}
        else:
            dispatcher.utter_message(response="utter_wrong_remove_resources_role")
            for resource in resources:
                dispatcher.utter_message(resource)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"remove_resources_role": None}

    def validate_remove_resources_transfer_role(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate remove_resources_transfer_role value."""

        resources = self.remove_resources_role_db(tracker)

        if value.lower() in [x.lower() for x in resources]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"remove_resources_transfer_role": value}
        else:
            dispatcher.utter_message(response="utter_wrong_remove_resources_transfer_role")
            for resource in resources:
                dispatcher.utter_message(resource)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"remove_resources_transfer_role": None}

class RemoveResourcesForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "remove_resources_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["remove_resources_role", "remove_resources_transfer_role"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class AskForRemoveResourceRole(Action):
    def name(self) -> Text:
        return "action_ask_remove_resources_role"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which resource do you want to remove?")
        
        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)
        for resource in df_resources['resourceName']:
            dispatcher.utter_message(text=resource)
        
        return []

class AskForRemoveResourceTransferRole(Action):
    def name(self) -> Text:
        return "action_ask_remove_resources_transfer_role"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="To which resource do you want to reallocate the removed resource tasks?")
        
        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)
        for resource in df_resources['resourceName']:
            if resource != tracker.get_slot("remove_resources_role"):
                dispatcher.utter_message(text=resource)        
        return []

class ActionRemoveResources(Action):
    def name(self) -> Text:
        return "action_remove_resources"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        df_resources = u.extract_resources(model_path)
        df_timetables = u.extract_timetables(model_path)
        
        res_remove = tracker.get_slot("remove_resources_role")
        new_res_remove = tracker.get_slot("remove_resources_transfer_role")

        df_elements = u.extract_elements(model_path)
        df_tasks = u.extract_tasks(model_path)
        df_tasks_elements = df_tasks.merge(df_elements, how='left', on='elementId')
        df = df_tasks_elements[['taskName', 'elementId', 'resourceId']].merge(df_resources, how='left', on='resourceId')

        resource = df_resources[df_resources['resourceName'] == res_remove][['resourceId', 'resourceName']]
        new_resource = df_resources[df_resources['resourceName'] == new_res_remove ][['resourceId', 'resourceName']]
        
        ptt_s = '<qbp:elements>'
        ptt_e = '</qbp:elements>'
        elements = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)
        element_lines = elements.split('\n')
        elements_list = []
        start, end = None, None
        for idx, line in enumerate(element_lines):
            if '<qbp:element ' in line and start == None:
                start = idx
            if '</qbp:element>' in line and end == None:
                end = idx
            if start != None and end != None:
                elements_list.append('\n'.join(element_lines[start:end+1]))
                start, end = None, None
                
        # Extract new elements and replace old one with new elements extracted
        new_elements = []
        for i in range(len(elements_list)):
            element = elements_list[i]
            old_e = list(resource['resourceId'])[0]
            new_e = list(new_resource['resourceId'])[0]
            if '<qbp:resourceId>{}</qbp:resourceId>'.format(list(resource['resourceId'])[0]) in element:
                new_element = element.replace(old_e, new_e)
            else:
                new_element = element
            new_elements.append(new_element)
        
        new_elements = '\n'.join([element_lines[0]] + new_elements + [element_lines[-1]])
        with open(model_path) as file:
            model= file.read()
        new_model = model.replace(elements, new_elements) 
        
        ptt_s = '<qbp:resources>'
        ptt_e = '</qbp:resources>'
        resources = u.extract_bpmn_resources(model_path, ptt_s, ptt_e).split('\n')
        new_resources = '\n'.join([x for x in resources if 'name="{}"'.format(list(resource['resourceName'])[0]) not in x])
        new_model = new_model.replace('\n'.join(resources), new_resources)
        
        new_model_path = model_path.split('.')[0] + '_rem_resource_{}'.format(res_remove.replace(' ', '_')) + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/resources/models')
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
        
        sce_name = 'Remotion of resource {}'.format(res_remove)
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_rem_resource_{}.csv'.format(res_remove.replace(' ', '_'))
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Remove resource')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/resources/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)

        return [SlotSet("remove_resources_role", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("remove_resources_transfer_role", None)]

class ValidateCreateWorkingTimeForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_create_working_time_form"

    @staticmethod
    def create_working_time_name_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_timetables = u.extract_timetables(model_path)

        return list(df_timetables['timetableName'])

    @staticmethod
    def create_working_time_resource_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)

        return list(df_resources['resourceName'])

    @staticmethod
    def create_working_time_weekday_db() -> List[Text]:
        """Database of supported roles for working times."""

        return ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""
        try:
            int(string)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_hour(string: Text) -> bool:
        """Check if a string have hour time format"""
        try:
            datetime.strptime(string, '%H:%M:%S')
            return True
        except:
            return False

    def validate_create_working_time_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_name value."""

        timetables = self.create_working_time_name_db(tracker)

        if value.lower() not in [x.lower() for x in timetables]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"create_working_time_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_name")
            for timetable in timetables:
                dispatcher.utter_message(timetable)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"create_working_time_name": None}

    def validate_create_working_time_from_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_from_time value."""

        if self.is_hour(value):
            return {"create_working_time_from_time": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_from_time")
            # validation failed, set slot to None
            return {"create_working_time_from_time": None}

    def validate_create_working_time_resource(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_resource value."""

        resources = self.create_working_time_resource_db(tracker)

        if value.lower() in [x.lower() for x in resources]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"create_working_time_resource": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_resource")
            for resource in resources:
                dispatcher.utter_message(resource)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"create_working_time_resource": None}

    def validate_create_working_time_to_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_to_time value."""

        if self.is_hour(value):
            return {"create_working_time_to_time": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_to_time")
            # validation failed, set slot to None
            return {"create_working_time_to_time": None}

    def validate_create_working_time_from_weekday(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_from_weekday value."""

        weekdays = self.create_working_time_weekday_db()

        if value.lower() in [x.lower() for x in weekdays]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"create_working_time_from_weekday": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_from_weekday")
            for weekday in weekdays:
                dispatcher.utter_message(weekday)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"create_working_time_from_weekday": None}

    def validate_create_working_time_to_weekday(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate create_working_time_to_weekday value."""

        weekdays = self.create_working_time_weekday_db()

        if value.lower() in [x.lower() for x in weekdays]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"create_working_time_to_weekday": value}
        else:
            dispatcher.utter_message(response="utter_wrong_create_working_time_to_weekday")
            for weekday in weekdays:
                dispatcher.utter_message(weekday)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"create_working_time_to_weekday": None}

class CreateWorkingTimeForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "create_working_time_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["create_working_time_id", "create_working_time_name", "create_working_time_from_time", "create_working_time_to_time", 
        "create_working_time_from_weekday", "create_working_time_to_weekday", "create_working_time_resource"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class AskForCreateWorkingTimeResource(Action):
    def name(self) -> Text:
        return "action_ask_create_working_time_resource"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="To which resource do you want to apply the new timetable?")
        
        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)
        for resource in df_resources['resourceName']:
            dispatcher.utter_message(text=resource)
        
        return []

class ActionCreateWorkingTime(Action):
    def name(self) -> Text:
        return "action_create_working_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")
  
        ptt_s = '<qbp:timetables>'
        ptt_e = '</qbp:timetables>'
        time_tables_text = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)
        time_tables = time_tables_text.split('\n')
        
        data = []
        start = None
        end = None
        for idx, time_table in enumerate(time_tables):
            start = 0
            if '<qbp:timetable ' in time_table and start == None:
                start = idx
            elif '</qbp:timetable>' in time_table and end == None:
                end = idx
                data.append(time_tables[start:end+1])
                start, end = None, None
        
        df_tt = pd.DataFrame(data = [], columns = ['id','name','fromTime', 'toTime', 'fromWeekDay', 'toWeekDay'])
        ptts = ['id','name','fromTime', 'toTime', 'fromWeekDay', 'toWeekDay']
        for time_table in data:
            rules = []
            for line in time_table:
                row = {}
                for ptt in ptts:
                    ptt_s = r'{}="(.*?)"'.format(ptt)
                    text = re.search(ptt_s, line)
                    if ptt == 'id' and text != None:
                        id_tt = text.group(1)
                    elif ptt == 'name' and text != None:
                        name_tt = text.group(1)
                    elif text != None:
                        row[ptt] = text.group(1)
                if row != {}:
                    rules.append(row)
            df = pd.DataFrame(rules)
            df['id'] = id_tt
            df['name'] = name_tt
            df = df[ptts]
            df_tt = pd.concat([df_tt, df])

        scenario_name = []
        tt_id = tracker.get_slot("create_working_time_id")
        tt_name = tracker.get_slot("create_working_time_name")

        scenario_name.append('add_{}'.format(tt_name))

        new_tt_rules = []
        from_time = tracker.get_slot("create_working_time_from_time")
        to_time = tracker.get_slot("create_working_time_to_time")
        from_weekday = tracker.get_slot("create_working_time_from_weekday").upper()
        to_weekday = tracker.get_slot("create_working_time_to_weekday").upper()
        rule = [from_time, to_time, from_weekday, to_weekday]
        new_tt_rules.append(rule)

        new_tt_df = pd.DataFrame(new_tt_rules, columns = ['fromTime', 'toTime', 'fromWeekDay', 'toWeekDay'])
        new_tt_df['id'] = tt_id
        new_tt_df['name'] = tt_name

        df_tt = pd.concat([df_tt, new_tt_df[df_tt.columns]])
        
        ptt_s = '<qbp:resources>'
        ptt_e = '</qbp:resources>'
        resources_text = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)
        resources = resources_text.split('\n')
        
        ptts = ['id', 'name', 'totalAmount', 'costPerHour', 'timetableId']
        data = []
        for line in resources:
            row = {}
            for ptt in ptts:
                ptt_s = r'{}="(.*?)"'.format(ptt)
                text = re.search(ptt_s, line)
                if text != None:
                    row[ptt] = text.group(1)
            if row != {}:
                data.append(row)
                
        df_resources = pd.DataFrame(data)

        res_change_tt = tracker.get_slot("create_working_time_resource")

        df_resources.loc[df_resources['name'] == res_change_tt, 'timetableId'] = tt_id

        format_time_tables = """    <qbp:timetables>{}</qbp:timetables>"""
    
        format_time_table = """\n        <qbp:timetable id="{}" default="false" name="{}">
                <qbp:rules>{}</qbp:rules>
            </qbp:timetable>"""
        
        format_rules_time_tables = """\n            <qbp:rule fromTime="{}" toTime="{}" fromWeekDay="{}" toWeekDay="{}"/>"""
        
        time_tables = list(df_tt['name'].drop_duplicates())
        
        time_tables_updated = []
        for time_table in time_tables:
            df_time_table = df_tt[df_tt['name'] == time_table]
            name_tt = df_time_table['name'].values[0]
            id_tt = df_time_table['id'].values[0]
            df_time_table['rule'] = df_time_table.apply(lambda x: format_rules_time_tables.format(x['fromTime'], x['toTime'], x['fromWeekDay'], x['toWeekDay']), axis= 1)
            rules = """""".join(df_time_table['rule'])
            time_table_tmp = format_time_table.format(id_tt, name_tt, rules)
            time_tables_updated.append(time_table_tmp)
            
        final_time_tables = format_time_tables.format("""""".join(time_tables_updated))

        with open(model_path) as f:
            model = f.read()
            
        new_model = model.replace(time_tables_text, final_time_tables)

        resources = """    <qbp:resources>
          {} 
        </qbp:resources>"""
        
        resource = """<qbp:resource id="{}" name="{}" totalAmount="{}" costPerHour="{}" timetableId="{}"/>"""
        df_resources['resource'] = df_resources.apply(lambda x: resource.format(x['id'], x['name'], x['totalAmount'], \
                                                                                x['costPerHour'], x['timetableId']
                                                                                ), axis=1)
        new_resources = resources.format("""""".join(df_resources['resource']))
        
        new_model = new_model.replace(resources_text, new_resources)
    
        sce_name = '_' + ('_'.join(scenario_name)).replace('/', '_')
        new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/working_tables/models')
        
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
            
        csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/working_tables/output{}.csv'.format(sce_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Timetable Creation')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/working_tables/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)
        
        return [SlotSet("create_working_time_id", None),
                SlotSet("create_working_time_name", None),
                SlotSet("create_working_time_from_time", None),
                SlotSet("create_working_time_to_time", None),
                SlotSet("create_working_time_from_weekday", None),
                SlotSet("create_working_time_resource", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("create_working_time_to_weekday", None)]

class ValidateModifyWorkingTimeForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_modify_working_time_form"

    @staticmethod
    def modify_working_time_name_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_timetables = u.extract_timetables(model_path)

        return list(df_timetables['timetableName'])

    @staticmethod
    def modify_working_time_resource_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_resources = u.extract_resources(model_path)

        return list(df_resources['resourceName'])

    @staticmethod
    def modify_working_time_weekday_db() -> List[Text]:
        """Database of supported roles for working times."""

        return ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""
        try:
            int(string)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_hour(string: Text) -> bool:
        """Check if a string have hour time format"""
        try:
            datetime.strptime(string, '%H:%M:%S')
            return True
        except:
            return False

    def validate_modify_working_time_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate modify_working_time_name value."""

        timetables = self.modify_working_time_name_db(tracker)

        if value.lower() in [x.lower() for x in timetables]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"modify_working_time_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_modify_working_time_name")
            for timetable in timetables:
                dispatcher.utter_message(timetable)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"modify_working_time_name": None}

    def validate_modify_working_time_from_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate modify_working_time_from_time value."""

        if self.is_hour(value):
            return {"modify_working_time_from_time": value}
        else:
            dispatcher.utter_message(response="utter_wrong_modify_working_time_from_time")
            # validation failed, set slot to None
            return {"modify_working_time_from_time": None}

    def validate_modify_working_time_to_time(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate modify_working_time_to_time value."""

        if self.is_hour(value):
            return {"modify_working_time_to_time": value}
        else:
            dispatcher.utter_message(response="utter_wrong_modify_working_time_to_time")
            # validation failed, set slot to None
            return {"modify_working_time_to_time": None}

    def validate_modify_working_time_from_weekday(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate modify_working_time_from_weekday value."""

        weekdays = self.modify_working_time_weekday_db()

        if value.lower() in [x.lower() for x in weekdays]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"modify_working_time_from_weekday": value}
        else:
            dispatcher.utter_message(response="utter_wrong_modify_working_time_from_weekday")
            for weekday in weekdays:
                dispatcher.utter_message(weekday)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"modify_working_time_from_weekday": None}

    def validate_modify_working_time_to_weekday(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate modify_working_time_to_weekday value."""

        weekdays = self.modify_working_time_weekday_db()

        if value.lower() in [x.lower() for x in weekdays]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"modify_working_time_to_weekday": value}
        else:
            dispatcher.utter_message(response="utter_wrong_modify_working_time_to_weekday")
            for weekday in weekdays:
                dispatcher.utter_message(weekday)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"modify_working_time_to_weekday": None}

class ModifyWorkingTimeForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "modify_working_time_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["modify_working_time_name", "modify_working_time_from_time", "modify_working_time_to_time", 
        "modify_working_time_from_weekday", "modify_working_time_to_weekday"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """

        return []

class AskForModifyWorkingTimeName(Action):
    def name(self) -> Text:
        return "action_ask_modify_working_time_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        dispatcher.utter_message(text="Which timetable do you want to modify?")

        model_path = tracker.get_slot("model")
        df_timetables = u.extract_timetables(model_path)
        for timetable in df_timetables['timetableName']:
            dispatcher.utter_message(text=timetable)

        return []



class ActionModifyWorkingTime(Action):
    def name(self) -> Text:
        return "action_modify_working_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        ptt_s = '<qbp:timetables>'
        ptt_e = '</qbp:timetables>'
        time_tables_text = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)
        time_tables = time_tables_text.split('\n')
        
        data = []
        start = None
        end = None
        for idx, time_table in enumerate(time_tables):
            start = 0
            if '<qbp:timetable ' in time_table and start == None:
                start = idx
            elif '</qbp:timetable>' in time_table and end == None:
                end = idx
                data.append(time_tables[start:end+1])
                start, end = None, None
        
        df_tt = pd.DataFrame(data = [], columns = ['id','name','fromTime', 'toTime', 'fromWeekDay', 'toWeekDay'])
        ptts = ['id','name','fromTime', 'toTime', 'fromWeekDay', 'toWeekDay']
        for time_table in data:
            rules = []
            for line in time_table:
                row = {}
                for ptt in ptts:
                    ptt_s = r'{}="(.*?)"'.format(ptt)
                    text = re.search(ptt_s, line)
                    if ptt == 'id' and text != None:
                        id_tt = text.group(1)
                    elif ptt == 'name' and text != None:
                        name_tt = text.group(1)
                    elif text != None:
                        row[ptt] = text.group(1)
                if row != {}:
                    rules.append(row)
            df = pd.DataFrame(rules)
            df['id'] = id_tt
            df['name'] = name_tt
            df = df[ptts]
            df_tt = pd.concat([df_tt, df])


        scenario_name = []
        change_tt_name = tracker.get_slot("modify_working_time_name")
        change_tt_df = df_tt[df_tt['name'].str.lower() == change_tt_name.lower()]
        
        df_tt = df_tt[df_tt['name'] != change_tt_name]
        scenario_name.append('modify_{}'.format(change_tt_name))
        
        tt_id = change_tt_df['id'].drop_duplicates().values[0]
        tt_name = change_tt_df['name'].drop_duplicates().values[0]

        new_tt_rules = []
        from_time = tracker.get_slot("modify_working_time_from_time")
        to_time = tracker.get_slot("modify_working_time_to_time")
        from_weekday = tracker.get_slot("modify_working_time_from_weekday").upper()
        to_weekday = tracker.get_slot("modify_working_time_to_weekday").upper()
        rule = [from_time, to_time, from_weekday, to_weekday]
        new_tt_rules.append(rule)

        new_tt_df = pd.DataFrame(new_tt_rules, columns = ['fromTime', 'toTime', 'fromWeekDay', 'toWeekDay'])
        new_tt_df['id'] = tt_id
        new_tt_df['name'] = tt_name
        df_tt = pd.concat([df_tt, new_tt_df[df_tt.columns]])

        format_time_tables = """    <qbp:timetables>{}</qbp:timetables>"""
    
        format_time_table = """\n        <qbp:timetable id="{}" default="false" name="{}">
                <qbp:rules>{}</qbp:rules>
            </qbp:timetable>"""
        
        format_rules_time_tables = """\n            <qbp:rule fromTime="{}" toTime="{}" fromWeekDay="{}" toWeekDay="{}"/>"""
        
        time_tables = list(df_tt['name'].drop_duplicates())
        
        time_tables_updated = []
        for time_table in time_tables:
            df_time_table = df_tt[df_tt['name'] == time_table]
            name_tt = df_time_table['name'].values[0]
            id_tt = df_time_table['id'].values[0]
            df_time_table['rule'] = df_time_table.apply(lambda x: format_rules_time_tables.format(x['fromTime'], x['toTime'], x['fromWeekDay'], x['toWeekDay']), axis= 1)
            rules = """""".join(df_time_table['rule'])
            time_table_tmp = format_time_table.format(id_tt, name_tt, rules)
            time_tables_updated.append(time_table_tmp)
            
        final_time_tables = format_time_tables.format("""""".join(time_tables_updated))

        with open(model_path) as f:
            model = f.read()
            
        new_model = model.replace(time_tables_text, final_time_tables)

        sce_name = ('_' + '_'.join(scenario_name)).replace('/', '_')
        new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
        new_model_path = new_model_path.replace('inputs','inputs/working_tables/models')
        
        with open(new_model_path, 'w+') as new_file:
            new_file.write(new_model)
            
        csv_output_path = 'C:/CursosMaestria/Tesis/Chatbot/outputs/working_tables/output{}.csv'.format(sce_name)
        u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
        output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Timetable Modification')
        
        csv_org_path = 'C:/CursosMaestria/Tesis/Chatbot/outputs/working_tables/output_baseline.csv'
        u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
        org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

        dispatcher.utter_message(text=org_message)
        dispatcher.utter_message(text=output_message)
        
        return [SlotSet("modify_working_time_name", None),
                SlotSet("modify_working_time_from_time", None),
                SlotSet("modify_working_time_to_time", None),
                SlotSet("modify_working_time_from_weekday", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("modify_working_time_to_weekday", None)]

class ValidateAutomateTaskForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_automate_task_form"

    @staticmethod
    def automate_task_name_db(tracker) -> List[Text]:
        """Database of supported resource timetables."""

        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)

        return list(df_tasks['taskName'])

    @staticmethod
    def is_int(string: Text) -> bool:
        """Check if a string is an integer."""
        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate_automate_task_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate automate_task_name value."""

        tasks = self.automate_task_name_db(tracker)

        if value.lower() in [x.lower() for x in tasks]:
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"automate_task_name": value}
        else:
            dispatcher.utter_message(response="utter_wrong_automate_task_name")
            for task in tasks:
                dispatcher.utter_message(task)
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"automate_task_name": None}

    def validate_automate_task_percentage(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate automate_task_percentage value."""

        if self.is_int(value) and int(value)>0 and int(value)<=100 :
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"automate_task_percentage": value}
        else:
            dispatcher.utter_message(response="utter_wrong_automate_task_percentage")
            # validation failed, set this slot to None, meaning the
            # user will be asked for the slot again
            return {"automate_task_percentage": None}

   
class AutomateTaskForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "automate_task_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["automate_task_name","automate_task_percentage"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """
        return []

class AskForAutomateTaskName(Action):
    def name(self) -> Text:
        return "action_ask_automate_task_name"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which task do you want to automate?")
        
        model_path = tracker.get_slot("model")
        df_tasks = u.extract_tasks(model_path)
        for task in df_tasks['taskName']:
            dispatcher.utter_message(text=task)
        
        return []

class ActionAutomateTask(Action):
    def name(self) -> Text:
        return "action_automate_task"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        with open(model_path) as file:
            model= file.read()

        df_tasks, _ = u.extract_task_add_info(model_path)

        task = tracker.get_slot("automate_task_name")
        percentage = int(tracker.get_slot("automate_task_percentage"))/100

        df_tasks_new = df_tasks.copy()

        if percentage == 100:
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['type']] = 'UNIFORM'
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['mean']] = 0
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['arg1']] = 0.0
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['arg2']] = 0.0
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['resourceName']] = 'SYSTEM'
        else:
            df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['mean']] = (1-percentage)*df_tasks_new.loc[df_tasks_new['name'].str.lower() == task.lower(), ['mean']]

            resource_msg = """      <qbp:element id="{}" elementId="{}">
                    <qbp:durationDistribution type="{}" mean="{}" arg1="{}" arg2="{}">
                    <qbp:timeUnit>{}</qbp:timeUnit>
                    </qbp:durationDistribution>
                    <qbp:resourceIds>
                    <qbp:resourceId>{}</qbp:resourceId>
                    </qbp:resourceIds>
                </qbp:element>"""

            elements_new = '\n'.join([resource_msg.format(x['id'], x['elementId'], x['type'], x['mean'], \
                        x['arg1'], x['arg2'], x['timeUnit'], x['resourceId']) for idx, x in df_tasks_new.iterrows()])

            elements_new = """    <qbp:elements>
            {}
    </qbp:elements>""".format(elements_new)
            ptt_s = '<qbp:elements>'
            ptt_e = '</qbp:elements>'
            elements_old = u.extract_bpmn_resources(model_path, ptt_s, ptt_e)

            new_model = model.replace(elements_old, elements_new)

            sce_name = '_automate_task_{}'.format(task.replace(' ', '_'))

            new_model_path = model_path.split('.')[0] + sce_name + '.bpmn'
            new_model_path = new_model_path.replace('inputs','inputs/automate_task/models')
            with open(new_model_path, 'w+') as new_file:
                new_file.write(new_model)
                
            csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/automate_task/output_{}.csv'.format(sce_name)
            u.execute_simulator_simple(bimp_path, new_model_path, csv_output_path)
            output_message = u.return_message_stats(csv_output_path, 'Stats for the what-if scenario: Task Automation')

            csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/automate_task/output_baseline.csv'
            u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
            org_message = u.return_message_stats(csv_org_path, 'Stats for the baseline scenario')

            dispatcher.utter_message(text=org_message)
            dispatcher.utter_message(text=output_message)
        
        return [SlotSet("automate_task_name", None),
                SlotSet("comparison_scenario", new_model_path),
                SlotSet("name_scenario", sce_name),
                SlotSet("automate_task_percentage", None)]

class AskMoreInformationForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "more_information_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["more_information"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """
        return []

class AskMoreInformation(Action):

    def name(self) -> Text:
        return "action_more_information"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'
        model_path = tracker.get_slot("model")

        ask_more_info = tracker.get_slot("more_information")
        comparison_scenario = tracker.get_slot("comparison_scenario")
        name_scenario = tracker.get_slot("name_scenario")

        if ask_more_info.lower() == 'yes':
            csv_output_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/automate_task/output_{}.csv'.format(name_scenario)
            u.execute_simulator_simple(bimp_path, comparison_scenario, csv_output_path)
            output_message = u.return_message_stats_complete(csv_output_path, '{}'.format(' '.join(str(name_scenario).split('_'))))

            csv_org_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/outputs/automate_task/output_baseline.csv'
            u.execute_simulator_simple(bimp_path, model_path, csv_org_path)
            org_message = u.return_message_stats_complete(csv_org_path, 'Stats for the baseline scenario')

            dispatcher.utter_message(text=org_message)
            dispatcher.utter_message(text=output_message)

        return [SlotSet("more_information", None),
                SlotSet("comparison_scenario", None),
                SlotSet("name_scenario", None)]

class ValidateComparisonScenariosForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_compare_scenarios_form"

    @staticmethod
    def scenarios_db() -> List[Text]:
        """Database of supported scenarios."""

        dict_scenarios = u.extract_scenarios()
        return dict_scenarios

    def validate_compared_scenarios(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate compared_scenarios value."""

        scenarios = self.scenarios_db()
        scenarios_names = {x : scenarios[x].split('\\')[-1] for x in scenarios.keys()}
        input_scenarios = [int(x.strip()) for x in value.split(',') if x.strip() != '' and x.strip() != ' ' and x.strip().isdigit()]
        compared_scenarios = []

        if len(input_scenarios)==0:
            dispatcher.utter_message(response="utter_wrong_compared_scenarios")
            for scenario in scenarios_names.keys():
                dispatcher.utter_message(json.dumps({scenario: scenarios_names[scenario].split('\\')[-1].split('.')[0]}))
            return {"compared_scenario_names": None}
        else:
            for input_scenario in input_scenarios:

                if input_scenario in scenarios.keys():
                    # validation succeeded, set the value of the "cuisine" slot to value
                    compared_scenarios.append(scenarios[input_scenario])
            if len(compared_scenarios)>0:
                return {"compared_scenario_names": compared_scenarios}
            else:
                dispatcher.utter_message(response="utter_wrong_compared_scenarios")
                for scenario in scenarios_names.keys():
                    dispatcher.utter_message(json.dumps({scenario: scenarios_names[scenario].split('\\')[-1].split('.')[0]}))
                return {"compared_scenario_names": None}

class CompareScenariosForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "compare_scenarios_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["compared_scenarios"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """
        return []

class AskForComparedScenarios(Action):
    def name(self) -> Text:
        return "action_ask_compared_scenarios"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="Which of the following scenarios do you want to compare? (write the number of scenarios separated by coma i.e. 1,2,4)")
        
        scenarios = u.extract_scenarios()
        for scenario in scenarios.keys():
            dispatcher.utter_message(json.dumps({scenario: scenarios[scenario].split('\\')[-1].split('.')[0]}))
        
        return []

class CompareScenarios(Action):

    def name(self) -> Text:
        return "action_compare_scenarios"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        bimp_path = 'C:/CursosMaestria/Tesis/What-If-Chatbot/bimp/qbp-simulator-engine_with_csv_statistics.jar'

        compared_scenarios = tracker.get_slot("compared_scenario_names")

        scenario_paths = []
        for comparison_scenario in compared_scenarios:
            name_sce = comparison_scenario.split('\\')[-1].split('.')[0].replace('_', ' ')
            csv_output_path = 'outputs/comparison/' + comparison_scenario.replace('bpmn', 'csv').split('\\')[-1]
            u.execute_simulator_simple(bimp_path, comparison_scenario, csv_output_path)
            scenario_message = u.return_message_stats_complete(csv_output_path, name_sce)
            dispatcher.utter_message(text=scenario_message)

        return [SlotSet("compared_scenarios", None),
                SlotSet("compared_scenario_names", None)]

class AskForModel(Action):
    def name(self) -> Text:
        return "action_ask_model"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        
        dispatcher.utter_message(text="On which of these models?")

        models = {idx: x for idx, x in enumerate(glob('inputs/*.bpmn'), start=1)}
        
        for key in models.keys():
            dispatcher.utter_message(text=json.dumps({key : models[int(key)]}))
        
        return []

class ValidateChooseModelForm(FormValidationAction):

    def name(self) -> Text:
        return "validate_choose_model_form"

    @staticmethod
    def models_db() -> List[Text]:
        """Database of supported models."""

        models = {idx: x for idx, x in enumerate(glob('inputs/*.bpmn'), start=1)}
        return models

    def validate_model(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate compared_scenarios value."""

        models = self.models_db()
        try:
            if int(value) in models.keys():
                return {"model": models[int(value)]}
            else:
                dispatcher.utter_message(text='Please enter a valid option for model')
                for key in models.keys():
                    dispatcher.utter_message(text=json.dumps({key : models[key]}))
                return {"model": None}
        except:
            return {"model": value}

class ChooseModelForm(FormAction):

    def name(self):
        """Unique identifier of the form"""
        return "choose_model_form"

    def required_slots(tracker: Tracker) -> List[Text]:
        """A list of required slots that the form has to fill"""

        return ["model"]

    def submit(self):
        """
        Define what the form has to do
        after all required slots are filled
        """
        return []