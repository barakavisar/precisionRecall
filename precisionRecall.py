import numpy as np
import pandas as pd


def mAP_calc(eval_table):
    prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 300):
        try:
            x = eval_table[eval_table['Recall'] >= recall_level]['Precision']
            #print('x', x)
            prec = max(x)
        except:
            prec = 0.0
        prec_at_rec.append(prec)
    mean_prec = np.mean(prec_at_rec)
    print('300 points precision is ', prec_at_rec)
    print('mAP is ', mean_prec)
    return mean_prec



def precision_recall(merged_data, pr_table_path):

    Precision = []
    Recall = []

    # initializing
    TP, FP, FN =  0, 0, 0
    # creating a evaluation table
    len_FN = len(merged_data[merged_data['iou'] == -11])
    merged_data = merged_data[merged_data['iou'] != -11].sort_values(['detection_prob'], ascending=False)
    print('len FN', len_FN)
    #print(merged_data.head())
    eval_table = pd.DataFrame()
    eval_table['image_name'] = merged_data['detection_filename'].values
    eval_table['detection_prob'] = merged_data['detection_prob'].values
     # Just for initiating
    len_table = len(eval_table)

    # creating column 'TP/FP' which will store TP for True positive and FP for False positive
    # if IOU is greater than 0.5 then TP else FP

    merged_data_iou = merged_data['iou'].values
    eval_table_TP_FP = merged_data['detection_filename'].values

    for k in range(len_table):
        if merged_data_iou[k] >= 0.33:
            eval_table_TP_FP[k] = 'TP'
        #if merged_data['iou'][k]  == -11:
        #    eval_table['TP/FP'][k] = 'FN'
        if merged_data_iou[k] == -111 or merged_data_iou[k] == -112:
            eval_table_TP_FP[k] = 'FP'
    eval_table['TP/FP'] = eval_table_TP_FP #merged_data['detection_filename'].values
    
    # assuming that we have a very bad model which misclassified all the objects. so all true positive becomes false negative
    #all_FN = len(eval_table['TP/FP'] == 'FN')
    all_TP = len(eval_table[eval_table['TP/FP'] == 'TP'])
    #print('all TP', all_TP)
    all_ground_truth = len_FN + all_TP
    for index, row in merged_data.iterrows():

        if row.iou >= 0.33:
            TP = TP + 1
        #elif row.iou == -11:
        #    FN = FN + 1
        #    print('false negative', FN)
        if row.iou == -111 or row.iou == -112:
            #print('false positive', FP)
            FP = FP + 1
        try:

            AP = TP / (TP + FP)
            Rec = TP / all_ground_truth #(TP + FN)

        except ZeroDivisionError:

                    AP = 0.0
                    Rec = 0.0
        Precision.append(AP)
        Recall.append(Rec)

    eval_table['Precision'] = Precision
    eval_table['Recall'] = Recall

    # calculating Interpolated Precision
    eval_table['IP'] = eval_table.groupby('Recall')['Precision'].transform('max')
    pr_table = eval_table

    pr_table.to_csv(pr_table_path)

    mAP = mAP_calc(eval_table)
    #print('maP', mAP)
    return mAP #, pr_table

