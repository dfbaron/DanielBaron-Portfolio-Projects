import subprocess
import re
import pandas as pd
import os
from glob import glob

# =============================================================================
#           CHANGE DEMAND SCENARIOS
# =============================================================================
def execute_simulator_simple(bimp_path, model_path, csv_output_path):
    args = ['java', '-jar', bimp_path, model_path, '-csv', csv_output_path]
    subprocess.run(args, stdout=open(os.devnull, 'wb'))

def extract_bpmn_components(model_path, ptt_s, ptt_e):
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    start, end = None, None
    for idx, line in enumerate(lines):
        if ptt_s in line and start == None:
            start = idx
        if ptt_e in line and end == None:
            end = idx
        if start != None and end != None:
            break
    return '\n'.join(lines[start:end+1])

def modify_bimp_model_instances(path_bimp_model, inc_percentage):
    
    with open(path_bimp_model) as file:
        model_bimp = file.read()
    
    ptt = r'processInstances="(.*?)"'
    process_inst = int(re.search(ptt, model_bimp).group(1))
    new_instances = int(process_inst*(1+inc_percentage))

    rep_proc_ins = 'processInstances="{}"'.format(process_inst)
    new_rep_proc_ins = 'processInstances="{}"'.format(new_instances)
    model_bimp = model_bimp.replace(rep_proc_ins, new_rep_proc_ins)
    new_model_path = path_bimp_model.split('.')[0] + '_inst_{}'.format(new_instances) + '.bpmn'
    
    with open(new_model_path.replace('inputs','inputs/demand/models'), 'w+') as new_file:
        new_file.write(model_bimp)
    
    return new_model_path.replace('inputs','inputs/demand/models')

def modify_bimp_model_interarrival_time(path_bimp_model, inc_percentage):
    
    with open(path_bimp_model) as file:
        model_bimp = file.read()
    
    int_arr_time_s = '<qbp:arrivalRateDistribution'
    int_arr_time_e = '</qbp:arrivalRateDistribution>'
    arr_rate_dist = extract_bpmn_components(path_bimp_model, int_arr_time_s, int_arr_time_e)
    ptt = r'mean="(.*?)"'
    
    mean = int(re.search(ptt, arr_rate_dist).group(1))
    new_mean = int(mean*(1-inc_percentage))

    proc_mean = 'mean="{}"'.format(mean)
    new_proc_mean = 'mean="{}"'.format(new_mean)
    
    model_bimp = model_bimp.replace(proc_mean, new_proc_mean)
    new_model_path = path_bimp_model.split('.')[0] + '_arr_time_{}'.format(new_mean) + '.bpmn'
    
    with open(new_model_path.replace('inputs','inputs/demand/models'), 'w+') as new_file:
        new_file.write(model_bimp)
    
    return new_model_path.replace('inputs','inputs/demand/models')

def extract_text(model_path, ptt_s, ptt_e):
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    start, end = None, None
    for idx, line in enumerate(lines):
        if ptt_s in line and start == None:
            start = idx
        if ptt_e in line and end == None:
            end = idx
        if start != None and end != None:
            break
    return '\n'.join(lines[start+1:end])

def return_message_stats(stats_path, scenario_name):

    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    ptt_s = 'Scenario statistics'
    ptt_e = 'Process Cycle Time (s) distribution' 
    text = extract_text(stats_path, ptt_s, ptt_e)

    data = [x.split(',') for x in text.split('\n') if x != '']
    df = pd.DataFrame(data = data[1:], columns=data[0])

    df = df[df['KPI']== 'Process Cycle Time (s)']
    df['Average'] = df['Average'].astype(float).astype(str).apply(lambda x: format(float(x),".2f")).astype(float)
    df['Average'], df['Units'] = zip(*df.apply(lambda x: standarize_metric(x['Average'], x['KPI']), axis=1))
    df['Average'] = df['Average'].round(2)
    df['KPI'] = df.apply(lambda x: x['KPI'].replace(' (s)', ''), axis=1)


    message = '{}: \n'.format(scenario_name)
    message += '\n'.join(df['KPI'] + ': ' + df['Average'].astype(str) + ' ' + df['Units'])
    
    return message

def return_message_stats_complete(stats_path, scenario_name):
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    
    ptt_s = 'Scenario statistics'
    ptt_e = 'Process Cycle Time (s) distribution' 
    text = extract_text(stats_path, ptt_s, ptt_e)

    data = [x.split(',') for x in text.split('\n') if x != '']
    df = pd.DataFrame(data = data[1:], columns=data[0])

    df['Average'] = df['Average'].astype(float).astype(str).apply(lambda x: format(float(x),".2f")).astype(float)
    df['Average'], df['Units'] = zip(*df.apply(lambda x: standarize_metric(x['Average'], x['KPI']), axis=1))
    df['Average'] = df['Average'].round(2)
    df['KPI'] = df.apply(lambda x: x['KPI'].replace(' (s)', ''), axis=1)

    message = '{} \n'.format(scenario_name)
    message += '\n'.join(df['KPI'] + ': ' + df['Average'].astype(str) + ' ' + df['Units'])
    
    return message

def standarize_metric(value, kpi):
    if 'cost' not in kpi.lower():
        if (value <= 60*1.5):
            return value, 'seconds'
        elif (value > 60*1.5) and (value <= 60*60*1.5):
            return value/(60), 'minutes'
        elif (value > 60*60*1.5) and (value <= 60*60*24*1.5):
            return value/(60*60), 'hours'
        elif (value > 60*60*24*1.5) and (value <= 60*60*24*7*1.5):
            return value/(60*60*24), 'days'
        elif (value > 60*60*24*7*1.5):
            return value/(60*60*24*7), 'weeks'
    else:
        return value, ''

# =============================================================================
#               CHANGE RESOURCES SCENARIO
# =============================================================================

def extract_bpmn_resources(model_path, ptt_s, ptt_e):
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    start, end = None, None
    for idx, line in enumerate(lines):
        if ptt_s in line and start == None:
            start = idx
        if ptt_e in line and end == None:
            end = idx
        if start != None and end != None:
            break
    return '\n'.join(lines[start:end+1])

def modify_bimp_model_resources(model_path, amount, new_amount, cost, new_cost, mod_res):
    
    with open(model_path) as file:
        model_bimp = file.read()
    
    rep_res_amoount = 'totalAmount="{}"'.format(amount)
    new_rep_res_amoount = 'totalAmount="{}"'.format(new_amount)
    model_bimp = model_bimp.replace(rep_res_amoount, new_rep_res_amoount)
    
    rep_res_cost = 'costPerHour="{}"'.format(amount)
    new_rep_res_cost = 'costPerHour="{}"'.format(new_amount)
    model_bimp = model_bimp.replace(rep_res_cost, new_rep_res_cost)
    
    new_model_path = model_path.split('.')[0] + '_mod_resource_{}'.format(mod_res) + '.bpmn'
    
    with open(new_model_path.replace('inputs','inputs/resources/models'), 'w+') as new_file:
        new_file.write(model_bimp)
    
    return new_model_path.replace('inputs','inputs/resources/models')


def extract_resources(model_path):
    ptt_s = '<qbp:resources>'
    ptt_e = '</qbp:resources>'
    resources = extract_bpmn_resources(model_path, ptt_s, ptt_e).split('\n')
    
    ptts = ['id','name', 'totalAmount', 'costPerHour', 'timetableId']
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
            
    df = pd.DataFrame(data)
    df.columns = ['resourceId', 'resourceName', 'totalAmount', 'costPerHour', 'timetableId']
    return df    

def extract_elements(model_path):
    ptt_s = '<qbp:elements>'
    ptt_e = '</qbp:elements>'
    elements = extract_bpmn_resources(model_path, ptt_s, ptt_e).split('\n')
    elements_list = []
    start, end = None, None
    for idx, line in enumerate(elements):
        if '<qbp:element ' in line and start == None:
            start = idx
        if '</qbp:element>' in line and end == None:
            end = idx
        if start != None and end != None:
            elements_list.append(elements[start:end+1])
            start, end = None, None
    
    #Patterns next to extracted element
    ptts_n = ['id','elementId', 'type', 'mean', 'arg1', 'arg2']
    
    #Patterns between to extracted element
    ptts_b = [['<qbp:resourceId>','</qbp:resourceId>'],
              ['<qbp:timeUnit>', '</qbp:timeUnit>']]
    
    data_elements = []
    for line in elements_list:
        row = {}
        for elem in line:
            for ptt_n in ptts_n:
                ptt_s = r'{}="(.*?)"'.format(ptt_n)
                text = re.search(ptt_s, elem)
                if text != None:
                    row[ptt_n] = text.group(1)
            for ptt_b in ptts_b:
                ptt_s = r'{}(.*){}'.format(ptt_b[0], ptt_b[1])
                text = re.search(ptt_s, elem.lstrip().rstrip())
                if text != None:
                    row[ptt_b[0].split(':')[-1][:-1]] = text.group(1)
        if row != {}:
            data_elements.append(row)
    return pd.DataFrame(data_elements)

def extract_tasks(model_path):
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    tasks = []
    for line in lines:
        if 'task id' in line:
            tasks.append(line)
    ptts = ['id','name']
    data = []
    for task in tasks:
        row = {}
        for ptt in ptts:
            ptt_s = r'{}="(.*?)"'.format(ptt)
            text = re.search(ptt_s, task)
            if text != None:
                row[ptt] = text.group(1)
        if row != {}:
            data.append(row)
    df = pd.DataFrame(data)
    df.columns = ['elementId', 'taskName']
    return df

def extract_timetables(model_path):
    ptt_s = '<qbp:timetables>'
    ptt_e = '</qbp:timetables>'
    time_tables = extract_bpmn_resources(model_path, ptt_s, ptt_e).split('\n')
    ptts = ['id','name']
    data = []
    for time_table in time_tables:
        row = {}
        for ptt in ptts:
            ptt_s = r'{}="(.*?)"'.format(ptt)
            text = re.search(ptt_s, time_table)
            if text != None:
                row[ptt] = text.group(1)
        if row != {}:
            data.append(row)
    df = pd.DataFrame(data)
    df.columns = ['timetableId', 'timetableName']
    return df

def extract_task_add_info(model_path):
    
    with open(model_path) as file:
        model= file.read()
    lines = model.split('\n')
    tasks = []
    for line in lines:
        if 'task id' in line:
            tasks.append(line)
    ptts = ['id','name']
    data = []
    for task in tasks:
        row = {}
        for ptt in ptts:
            ptt_s = r'{}="(.*?)"'.format(ptt)
            text = re.search(ptt_s, task)
            if text != None:
                row[ptt] = text.group(1)
        if row != {}:
            data.append(row)
    df_tasks = pd.DataFrame(data)
    df_tasks.columns = ['elementId', 'name']    
    
    task_dist = []
    start = None
    end = None
    
    for idx, line in enumerate(lines):
        if '<qbp:elements>' in line and start == None:
            start = idx
        elif '</qbp:elements>' in line and end == None:
            end = idx
            task_dist.append(lines[start:end+1])
            break
    
    elements_taks = []
    start = None
    end = None
    
    for idx, line in enumerate(task_dist[0]):
        if '<qbp:element ' in line and start == None:
            start = idx
        elif '</qbp:element>' in line and end == None:
            end = idx
            elements_taks.append(task_dist[0][start:end+1])
            start, end = None, None
    
    ptts = ['id', 'elementId','type','mean', 'arg1', 'arg2', 'timeUnit', 'resourceId']
    tasks = []
    for task_elem in elements_taks:
        row = {}
        for task_line in task_elem:
            for ptt in ptts:
                ptt_s = r'{}="(.*?)"'.format(ptt)
                text = re.search(ptt_s, task_line)
                if text != None:
                    row[ptt] = text.group(1)
                elif ptt == 'timeUnit':
                    ptt_s = r'<qbp:timeUnit>(.*?)</qbp:timeUnit>'
                    text = re.search(ptt_s, task_line)
                    if text != None:
                        row[ptt] = text.group(1)
                elif ptt == 'resourceId':
                    ptt_s = r'<qbp:resourceId>(.*?)</qbp:resourceId>'
                    text = re.search(ptt_s, task_line)
                    if text != None:
                        row[ptt] = text.group(1)
        if row != {}:
            tasks.append(row)
    
    df_tasks_dist = pd.DataFrame(tasks)
    df_tasks_dist[['mean', 'arg1', 'arg2']] = df_tasks_dist[['mean', 'arg1', 'arg2']].astype(float)
    df_tasks = df_tasks.merge(df_tasks_dist, on='elementId', how='left')

    return df_tasks, task_dist

def extract_scenarios():
    scenarios = glob('inputs/*/models/*.bpmn')

    available_scenarios = [x for x in scenarios]
    df_available_scenarios = pd.DataFrame(data = available_scenarios, columns = ['SCENARIOS'])

    return dict(enumerate(list(df_available_scenarios['SCENARIOS']), start=1))