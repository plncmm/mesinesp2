def f1_score(real, pred):
    tp = 0
    fp = 0
    fn = 0
    total = 0
    for r, p in zip(real, pred):
        for tag in r:
            total+=1
            if tag in p:
                tp+=1
            else:
                fn+=1      
        for tag in p:
            if tag not in r:
                fp+=1
    recall = tp/(tp+fn) if (tp+fn)!=0 else 0
    precision = tp/(tp+fp) if (tp+fp)!=0 else 0
    return tp, fn, fp, recall, precision, (2*recall*precision)/(recall+precision)