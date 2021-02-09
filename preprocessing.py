import numpy as np
import pickle
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from itertools import chain


class Visit(object):
    def __init__(self, start_date, end_date, is_inpatient, is_emergency):
        self.start_date = start_date
        self.end_date = end_date
        self.is_inpatient = is_inpatient
        self.is_emergency = is_emergency
        self.codes = set()
        self.age = None
        # self.index = None
        self.interval = None


class Person(object):
    def __init__(self, gender, birth_date):
        self.visits = dict()
        self.gender = gender
        self.birth_date = birth_date


def build_dic():
    dic = dict()
    code2idx = dict()
    code_dic = dict()
    gender_dic = {'45880669':1,'45878463':0}

    data = pd.read_csv('2019_ccs_services_procedures.csv', skiprows=1, usecols=['Code Range', 'CCS'])
    code_range = data['Code Range'].values.astype(str)
    ccs = data['CCS'].values.astype(str)

    for x, y in zip(code_range, ccs):
        if 'PR' + y not in code2idx:
            code2idx['PR' + y] = len(code2idx)
        start, end = x.strip('\'').split('-')
        if start[0].isalpha():
            for i in range(int(start[1:]), int(end[1:]) + 1):
                code_dic[start[0] + str(i).zfill(len(start)-1)] = code2idx['PR' + y]
        elif start[-1].isalpha():
            for i in range(int(start[:-1]), int(end[:-1]) + 1):
                code_dic[str(i).zfill(len(start)-1) + start[-1]] = code2idx['PR' + y]
        else:
            for i in range(int(start), int(end) + 1):
                code_dic[str(i).zfill(len(start))] = code2idx['PR' + y]

    with open('AppendixBSinglePR.txt', 'r') as f:
        for line in f:
            if line[0].isnumeric():
                y = str(line.strip('\n').strip(' ').split(' ')[0])
                if 'PR' + y not in code2idx:
                    code2idx['PR' + y] = len(code2idx)
            elif line[0].isspace():
                x = line.strip('\n').strip(' ').split(' ')
                for s in x:
                    code_dic[s[:2] + '.' + s[2:]] = code2idx['PR' + y]

    data = pd.read_csv('ccs_pr_icd10pcs_2020_1.csv', usecols=['\'ICD-10-PCS CODE\'', '\'CCS CATEGORY\''])
    code_range = data['\'ICD-10-PCS CODE\''].values.astype(str)
    ccs = data['\'CCS CATEGORY\''].values.astype(str)
    for x, y in zip(code_range, ccs):
        x = x.strip('\'')
        y = y.strip('\'')
        if 'PR' + y not in code2idx:
            code2idx['PR' + y] = len(code2idx)
        code_dic[x] = code2idx['PR' + y]

    with open('AppendixASingleDX.txt', 'r') as f:
        for line in f:
            if line[0].isnumeric():
                y = str(line.strip('\n').strip(' ').split(' ')[0])
                if 'DX' + y not in code2idx:
                    code2idx['DX' + y] = len(code2idx)
            elif line[0].isspace():
                x = line.strip('\n').strip(' ').split(' ')
                for s in x:
                    code_dic[s[:3] + '.' + s[3:]] = code2idx['DX' + y]

    data = pd.read_csv('ccs_dx_icd10cm_2019_1.csv', usecols=['\'ICD-10-CM CODE\'', '\'CCS CATEGORY\''])
    code_range = data['\'ICD-10-CM CODE\''].values.astype(str)
    ccs = data['\'CCS CATEGORY\''].values.astype(str)
    for x, y in zip(code_range, ccs):
        x = x.strip('\'')
        y = y.strip('\'')
        if 'DX' + y not in code2idx:
            code2idx['DX' + y] = len(code2idx)
        if len(x) <= 3:
            code_dic[x] = code2idx['DX' + y]
        else:
            code_dic[x[:3]+'.'+x[3:]] = code2idx['DX' + y]

    np.save('code_dic',code_dic,allow_pickle=True)
    np.save('code2idx', code2idx, allow_pickle=True)
    print('code done')

    with open('person_2017.csv', 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(',')
            person_id = tokens[0]
            try:
                gender = gender_dic[tokens[1]]
            except:
                continue
            person = Person(gender,datetime.date(*[int(x) for x in tokens[2:]]))
            dic[person_id] = person
    print('person done')

    with open('visit_2017.csv', 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(',')
            person_id = tokens[0]
            visit_id = tokens[1]
            start_date = datetime.date(*[int(x) for x in tokens[2].split('-')])
            visit_type = tokens[-1]
            if not visit_type:
                continue
            is_inpatient = (visit_type == '9201')
            is_emergency = (visit_type == '9203')
            if tokens[3]:
                end_date = datetime.date(*[int(x) for x in tokens[3].split('-')])
            else:
                end_date = start_date
            if end_date < start_date:
                end_date, start_date = start_date, end_date
            if (end_date - start_date).days > 2 and not is_inpatient:
                end_date = start_date
            if person_id in dic:
                if visit_id not in dic[person_id].visits:
                    visit = Visit(start_date, end_date, is_inpatient, is_emergency)
                    dic[person_id].visits[visit_id] = visit
                else:
                    if end_date <= dic[person_id].visits[visit_id].end_date \
                            and end_date >= dic[person_id].visits[visit_id].start_date:
                        dic[person_id].visits[visit_id].start_date = min([start_date,
                                                                        dic[person_id].visits[visit_id].start_date])
                    elif start_date >= dic[person_id].visits[visit_id].start_date \
                            and start_date <= dic[person_id].visits[visit_id].end_date:
                        dic[person_id].visits[visit_id].end_date = max([end_date,
                                                                        dic[person_id].visits[visit_id].end_date])
                    else:
                        visit = Visit(start_date, end_date, is_inpatient, is_emergency)
                        dic[person_id].visits[visit_id+'+'+start_date.isoformat()] = visit

    absentee = []
    with open('procedure_2017.csv', 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(',')
            if not tokens[-1]:
                continue
            tokens[-1] = tokens[-1].strip('"')
            if tokens[-1][0] != 'D':
                try:
                    code = code_dic[tokens[-1]]
                except:
                    absentee.append(tokens[-1])
                    continue
            else:
                continue
            person_id, visit_id, procedure_time = tokens[:-1]
            procedure_time = datetime.date(*[int(x) for x in procedure_time.split('-')])
            if person_id in dic:
                for vid in dic[person_id].visits:
                    if vid[:len(visit_id)] == visit_id:
                        if procedure_time <= dic[person_id].visits[vid].end_date \
                                and procedure_time >= dic[person_id].visits[vid].start_date:
                            dic[person_id].visits[vid].codes.add(code)
                            break

    print(set(absentee))
    print('procedure done')

    absentee = []
    with open('condition_2017.csv', 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            tokens = line.strip('\n').split(',')
            tokens[-1] = tokens[-1].strip('"')
            try:
                code = code_dic[tokens[-1]]
            except:
                if np.sum([x.isnumeric() for x in tokens[-1]]) > 0:
                    absentee.append(tokens[-1])
                continue
            person_id, visit_id, condition_time = tokens[:-1]
            condition_time = datetime.date(*[int(x) for x in condition_time.split('-')])
            if person_id in dic:
                for vid in dic[person_id].visits:
                    if vid[:len(visit_id)] == visit_id:
                        if condition_time <= dic[person_id].visits[vid].end_date \
                                and condition_time >= dic[person_id].visits[vid].start_date:
                            dic[person_id].visits[vid].codes.add(code)
                            break
    print('condition done')
    print(set(absentee))

    with open('patient_dic.pkl', 'wb') as f:
        pickle.dump(dic, f)

# ---------------------------------------------------------------------------------------------------

def merge(visits):
    i = 0
    j = len(visits) - 1
    while i < j:
        if visits[i].end_date >= visits[i + 1].start_date:
            visits[i].codes.update(visits[i + 1].codes)
            if visits[i].end_date < visits[i + 1].end_date:
                visits[i].end_date = visits[i + 1].end_date
            # visits[i].index = (visits[i].index or visits[i + 1].index)
            visits[i].is_emergency = (visits[i].is_emergency or visits[i + 1].is_emergency)
            visits[i].is_inpatient = (visits[i].is_inpatient or visits[i + 1].is_inpatient)
            visits.pop(i+1)
            j = len(visits) - 1
        else:
            i += 1
    return visits


def build_matrix(keys, threshold=5):
    code2idx = np.load('code2idx.npy', allow_pickle=True).item()
    dx_codes = set([code2idx[code] for code in code2idx.keys() if code.startswith('D')])
    print(len(dx_codes))

    def stay_dic(duration):
        if duration == 0:
            stay = 0
        elif duration == 1:
            stay = 1
        elif duration <= 2:
            stay = 2
        elif duration <= 4:
            stay = 3
        elif duration <= 6:
            stay = 4
        elif duration <= 14:
            stay = 5
        elif duration <= 30:
            stay = 6
        else:
            stay = 7
        return stay

    def age_dic(a):
        if a >= 80:
            return 0
        else:
            return int(a/5)+1

    targets = []
    histories = []
    codes = []
    others = []
    lengths = []
    ages = []
    male = 0
    weight = np.zeros(len(dx_codes))
    for patient_id in keys:
        patient = patient_dic[patient_id]
        visits = [visit for visit in patient.visits.values() if len(visit.codes) > 0]
        visits.sort(key=lambda x: x.start_date)
        visits = merge(visits)
        visits = [visit for visit in visits if visit.start_date >= datetime.date(2011,7,1)]
        length = len([visit for visit in visits if visit.end_date < datetime.date(2018,7,1) and
                      visit.start_date >= datetime.date(2016,7,1)])
        history_length = len([visit for visit in visits if visit.end_date < datetime.date(2018,7,1)])
        if history_length > 200:
            visits = visits[history_length-200:]
            history_length = 200
        post_length = len([visit for visit in visits[history_length:] if (visit.start_date-visits[history_length-1].end_date).days <= 180])
        if length < threshold or post_length == 0 or history_length < 25:
            continue
        history_codes = set([code for code in chain(*[visit.codes for visit in visits[:history_length]]) if code in dx_codes])
        if len(history_codes) == 0:
            continue
        for i in range(history_length):
            visits[i].duration = (visits[i].end_date - visits[i].start_date).days
            visits[i].age = int((visits[i].start_date - patient_dic[patient_id].birth_date).days / 365)
            if i == 0:
                visits[i].interval = 0
            else:
                interval = (visits[i].start_date - visits[i - 1].end_date).days
                if interval <= 0:
                    print(interval, patient_id)
                elif interval == 1:
                    visits[i].interval = 1
                elif interval == 2:
                    visits[i].interval = 2
                elif interval <= 4:
                    visits[i].interval = 3
                elif interval <= 6:
                    visits[i].interval = 4
                elif interval <= 14:
                    visits[i].interval = 5
                elif interval <= 30:
                    visits[i].interval = 6
                elif interval <= 90:
                    visits[i].interval = 7
                elif interval <= 180:
                    visits[i].interval = 8
                else:
                    visits[i].interval = 9

        labels = set()
        for visit in visits[history_length:history_length+post_length]:
            for code in visit.codes:
                if code in dx_codes and code not in history_codes:
                    labels.add(code)
        history_codes = np.array(list(history_codes)).astype('int')-244
        labels = np.array(list(labels))-244
        weight += 1
        weight[history_codes] -= 1

        # targets.append(labels)
        # histories.append(history_codes)
        targets.append(np.concatenate([labels, -np.ones(28-len(labels))],axis=-1))
        histories.append(np.concatenate([history_codes, -np.ones(115-len(history_codes))],axis=-1))
        lengths.append(history_length)
        tmp_codes = []
        tmp_others = []
        for visit in visits[:history_length]:
            tmp_codes.append(list(visit.codes))
            tmp_age = np.zeros(17)
            tmp_age[age_dic(visit.age)] = 1
            tmp_stay = np.zeros(8)
            tmp_stay[stay_dic(visit.duration)] = 1
            tmp_interval = np.zeros(10)
            tmp_interval[visit.interval] = 1
            tmp_gender = np.zeros(2)
            tmp_gender[patient.gender] = 1
            tmp_others.append(np.concatenate((tmp_age,tmp_gender,tmp_stay, tmp_interval,
                                              np.array([visit.is_emergency],dtype='int'),
                                              np.array([visit.is_inpatient],dtype='int'))))
        codes.append(tmp_codes)
        others.append(tmp_others)
        ages.append(visits[history_length-1].age)
        if patient.gender == 0:
            male += 1
    print(min(ages),np.median(ages),max(ages))
    print(male/len(ages))
    print('number of patient', len(targets))
    print('number of labels', np.max([len(x) for x in targets]))
    print('max_num_visit',np.max([len(x) for x in codes]))
    print('num_range',np.max([len(x) for x in histories]))
    print('max_length_visit',np.max([len(y) for x in codes for y in x]))

    # code_input = []
    # others_input = []
    # for code, other in zip(codes,others):
    #     code = [np.concatenate([np.array(x),-np.ones(112-len(x))],axis=-1) for x in code]
    #     code_input.append(np.concatenate((np.array(code),-np.ones((200-len(code), 112))),axis=0))
    #     others_input.append(np.concatenate((np.array(other),np.zeros((200-len(other), 39))),axis=0))
    #
    # np.save('length', np.array(lengths,dtype='int32'))
    # np.save('target', np.array(targets,dtype='int32'))
    # np.save('history', np.array(histories,dtype='int32'))
    # # np.save('weight', weight / np.sum(weight))
    # np.save('code',np.array(code_input,dtype='int32'))
    # np.save('others', np.array(others_input,dtype='int32'))


def split_chunk():
    targets = np.load('target.npy')
    train_idx, test_idx = train_test_split(np.arange(len(targets)), test_size=0.67)
    test_idx, val_idx = train_test_split(test_idx, test_size=0.5)
    np.save('train_idx',train_idx)
    np.save('test_idx',test_idx)
    np.save('val_idx',val_idx)


def label_weight():
    targets = np.load('target.npy')
    pos = np.zeros(283)
    for x in targets:
        x = x[x != -1]
        pos[x] += 1
    np.save('label_weight',pos)
    # print(np.sum(pos>=50))


if __name__ == '__main__':
    # build_dic()
    with open('patient_dic.pkl','rb') as file:
        patient_dic = pickle.load(file)
        print(len(patient_dic))
        build_matrix(patient_dic.keys())
    # split_chunk()
    # label_weight()




