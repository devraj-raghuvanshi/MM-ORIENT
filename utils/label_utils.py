import pandas as pd
import numpy as np

def get_train_labels(df, task, cat=0):
    """
    Generate training labels for different tasks and categories.
    
    Args:
        df: DataFrame containing the training data
        task: Task type ('A', 'B', or 'C')
        cat: Category for tasks B and C ('humour', 'sarcasm', 'offensive', 'motivational')
    
    Returns:
        numpy array of one-hot encoded labels
    """
    if task == 'A':
        def task_A_labels(dframe):
            def fun_A(text):
                if text=='positive' or text=='very_positive':
                    return 'positive'
                if text=='negative' or text=='very_negative':
                    return 'negative'
                return 'neutral'

            taskA_labels = dframe.overall_sentiment.apply(fun_A)
            return pd.get_dummies(taskA_labels)

        return np.array(task_A_labels(df)).astype(int)

    if task == 'B':
        if cat == 'humour':
            x = 'not_funny'

        elif cat == 'sarcasm':
            x = 'not_sarcastic'

        elif cat == 'offensive':
            x = 'not_offensive'

        elif cat == 'motivational':
            x = 'not_motivational'

        else:
            print('invalid cat')

        def func(text):
            if text == x:
                return 0
            return 1

        return np.array(pd.get_dummies(df[cat].apply(func))).astype(int)

    elif task == 'C':
        if cat == 'humour':
            labels_ohe = pd.get_dummies(df[cat])
            return np.array(labels_ohe[['not_funny', 'funny', 'very_funny', 'hilarious']]).astype(int)

        elif cat == 'sarcasm':
            labels_ohe = pd.get_dummies(df[cat])
            return np.array(labels_ohe[['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted']]).astype(int)

        elif cat == 'offensive':
            labels_ohe = pd.get_dummies(df[cat])
            return np.array(labels_ohe[['not_offensive', 'slight', 'very_offensive', 'hateful_offensive']]).astype(int)

        elif cat == 'motivational':
            def func(text):
                if text == 'not_motivational':
                    return 0
                return 1
            return np.array(pd.get_dummies(df[cat].apply(func))).astype(int)

        else:
            print('invalid category')
            return

def get_test_labels(df_test, task, cat=0):
    """
    Generate test labels for different tasks and categories.
    
    Task A: scores can be one of [-1, 0, 1]
    Task B: scores can be one of [0, 1] 
    Task C: scores can be one of [0, 1, 2, 3]
    
    The four digits for task-b and task-c are in the order: humor, sarcasm, offensive, motivational.
    
    Task B:
    - Not humorous => 0 and Humorous (funny, very funny, hilarious) => 1
    - Not Sarcastic => 0 and Sarcastic (general, twisted meaning, very twisted) => 1
    - Not offensive => 0 and Offensive (slight, very offensive, hateful offensive) => 1
    - Not Motivational => 0 and Motivational => 1
    
    Task C:
    Humour: Not funny => 0, Funny => 1, Very funny => 2, Hilarious => 3
    Sarcasm: Not Sarcastic => 0, General => 1, Twisted Meaning => 2, Very Twisted => 3
    Offense: Not offensive => 0, Slight => 1, Very Offensive => 2, Hateful Offensive => 3
    Motivation: Not Motivational => 0, Motivational => 1
    
    Args:
        df_test: DataFrame containing the test data
        task: Task type ('A', 'B', or 'C')
        cat: Category for tasks B and C ('humour', 'sarcasm', 'offensive', 'motivational')
    
    Returns:
        numpy array of one-hot encoded labels
    """
    if cat == 'humour':
        def fun(num):
            return int(num/1000)

    elif cat == 'sarcasm':
        def fun(num):
            return int((num%1000)/100)

    elif cat == 'offensive':
        def fun(num):
            return int((num%100)/10)

    elif cat == 'motivational':
        def fun(num):
            return int(num%10)
        if task=='C':
            df_test[cat+'_taskC'] = df_test['T3'].apply(fun)
            return np.array(pd.get_dummies(df_test['T3'].apply(fun))).astype(int)

    elif task=='A':
        return np.array(pd.get_dummies(df_test['T1'])).astype(int)

    else:
        print('invalid cat')
        return

    if task=='B':
        df_test[cat+'_taskB'] = df_test['T2'].apply(fun)
        return np.array(pd.get_dummies(df_test[cat+'_taskB'])).astype(int)

    elif task=='C':
        df_test[cat+'_taskC'] = df_test['T3'].apply(fun)
        return np.array(pd.get_dummies(df_test[cat+'_taskC'])).astype(int) 