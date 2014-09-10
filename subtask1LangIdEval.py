#!/usr/bin/python
# FIRE 2014 Shared Task on Transliterated Search
# Subtask 1/Language Identification
# Evaluation script for language identification
#
# Gokul Chittaranjan, gokulchittaranjan@gmail.com

import sys;
import re;

import numpy as np;

def readAnnotationFile(filename):
    nere = re.compile("\[[^\]]+\]");
    fid = open(filename,'r');
    data = [];
    for line in fid:
        line = line.replace("\n","").replace("\r","").decode('utf-8').strip();
        #print [line]
        if line=="":
            continue;
        neLocs = [];
        it = re.finditer(nere, line);
        for match in it:
            neLocs.append([match.start(), match.end()]);
        fields = line.split(" ");
        st = 0;
        entry = [];
        for f in fields:
            inNe = False;
            rloc = [];
            for loc in neLocs:
                if st>=loc[0] and st<=loc[1]:
                    rloc = loc;
                    inNe = True;
                    break;
            tok = "";
            tokType = "";
            translit = "";
            if inNe:
                tok = f;
                if rloc[0]==st:
                    tok = tok[1:];
                if rloc[1]==st+len(f)-1:
                    tok = tok[:-2];
                postfix = line[rloc[1]:];
                if len(postfix)==0:
                    tokType = "NE_GENERIC";
                elif postfix[0]==" ":
                    tokType = "NE_GENERIC";
                elif not postfix[0] in ["P","O","L","A"]:
                    tokType = "NE_?";
                else:
                    tokType = "NE_%s" %(postfix[0]);
                if len(postfix)>1:
                    temp = postfix.split(" "); # translit of multi-token ne not handled here...
                    if len(temp)>0:
                        if "=" in temp[0]:
                            translit = temp.split("=")[1];
            else:
                if "\\" in f:
                    temp = f.split("\\");
                    tok = temp[0];
                    if "=" in temp[1]:
                        translit = temp[1].split("=")[1];
                        tokType = temp[1].split("=")[0];
                    else:
                        tokType = temp[1];
                else:
                    tok = f;
            st += len(f)+1;
            entry.append([tok, tokType, translit]);
        data.append(entry);
    return data;
    

def printResult(results):
    
    outs = [];
    outs.append("Status\t%s" %(results["message"]));
    outs.append("Rows\t%d" %(results["rows"]));
    outs.append("Tokens\t%d" %(results["tokens"]));
    outs.append("Micro Average Accuracy\t%2.3f" %(results["micro_accuracy"]));
    outs.append("Macro Average Accuracy\t%2.3f" %(results["macro_accuracy"]));
    outs.append("Macro Average Precision\t%2.3f" %(results["p_macro"]));
    outs.append("Macro Average Recall\t%2.3f" %(results["r_macro"]));
    outs.append("Macro Average F-measure\t%2.3f" %(results["f_macro"]));
    for k in results["p_perClass"].keys():
        outs.append("Precision for class %s\t%2.3f" %(k, results["p_perClass"][k]));
        outs.append("Recall for class %s\t%2.3f" %(k, results["r_perClass"][k]));
        outs.append("F-measure for class %s\t%2.3f" %(k, results["f_perClass"][k]));
        outs.append("Accuracy for class %s\t%2.3f" %(k, results["acc_perClass"][k]));
        outs.append("Tokens for class %s\t%d" %(k, results["tokens_perClass"][k]));

    outs.append("No Mix: Macro Average Precision\t%2.3f" %(results["p_noMix"]));
    outs.append("No Mix: Average Recall\t%2.3f" %(results["r_noMix"]));
    outs.append("No Mix: Average F-measure\t%2.3f" %(results["f_noMix"]));
    outs.append("No Mix: Average Accuracy\t%2.3f" %(results["acc_noMix"]));
    outs.append("No Mix: Tokens\t%d" %(results["tokens_noMix"]));

    outs.append("No NE: Macro Average Precision\t%2.3f" %(results["p_noNe"]));
    outs.append("No NE: Average Recall\t%2.3f" %(results["r_noNe"]));
    outs.append("No NE: Average F-measure\t%2.3f" %(results["f_noNe"]));
    outs.append("No NE: Average Accuracy\t%2.3f" %(results["acc_noNe"]));
    outs.append("No NE: Tokens\t%d" %(results["tokens_noNe"]));

    outs.append("No Mix and NE: Macro Average Precision\t%2.3f" %(results["p_noMixAndNe"]));
    outs.append("No Mix and NE: Average Recall\t%2.3f" %(results["r_noMixAndNe"]));
    outs.append("No Mix and NE: Average F-measure\t%2.3f" %(results["f_noMixAndNe"]));
    outs.append("No Mix and NE: Average Accuracy\t%2.3f" %(results["acc_noMixAndNe"]));
    outs.append("No Mix and NE: Tokens\t%d" %(results["tokens_noMixAndNe"]));
    return "\n".join(outs);

def evaluateResult(gtfile, testfile):
    gtData = readAnnotationFile(gtfile);
    testData = readAnnotationFile(testfile);

    results = dict();
    results["micro_accuracy"] = -1;
    results["macro_accuracy"] = -1;
    results["p_macro"] = -1;
    results["r_macro"] = -1;
    results["f_macro"] = -1;
    
    results["p_perClass"] = dict();
    results["r_perClass"] = dict();
    results["f_perClass"] = dict();
    results["acc_perClass"] = dict();
    results["tokens_perClass"] = dict();
    
    results["p_noMix"] = -1;
    results["r_noMix"] = -1;
    results["f_noMix"] = -1;
    results["acc_noMix"] = -1;
    results["tokens_noMix"] = -1;

    results["p_noNe"] = -1;
    results["r_noNe"] = -1;
    results["f_noNe"] = -1;
    results["acc_noNe"] = -1;
    results["tokens_noNe"] = -1;


    results["p_noMixAndNe"] = -1;
    results["r_noMixAndNe"] = -1;
    results["f_noMixAndNe"] = -1;
    results["acc_noMixAndNe"] = -1;
    results["tokens_noMixAndNe"] = -1;
    
    results["rows"] = -1;
    results["tokens"] = -1;
    results["message"] = "No Results.";

    
    if len(gtData)!=len(testData):
        results["message"] = "GT and test files do not have the same number of rows.";
        return results;

    labels = [];
    for gtRow in gtData:
        for token in gtRow:
            if not token[1] in labels:
                labels.append(token[1]);
    for testRow in testData:
        for token in testRow:
            if not token[1] in labels:
                print "WARNING: Test data contained a label %s that was not in GT." %(token[1]);
                labels.append(token[1]);
    #if not "NE_" in labels:
    #    labels.append("NE_?");
    
    labels = sorted(labels);
    nLab = len(labels);
    
    rcnt = 0;
    confMatrix = np.zeros([nLab, nLab]);
    tot = 0;
    for gtRow, testRow in zip(gtData, testData):
        rcnt += 1;
        if len(gtRow) != len(testRow):
            results["message"] = "Row %s does not match." %(rcnt);
            return results;
        
        for gtTok, testTok in zip(gtRow, testRow):
            
            gtIdx = labels.index(gtTok[1]);
            if not testTok[1] in labels:
                results["message"] = "Test data contains label %s in row %s, which is not defined in GT" %(testTok[1], rcnt);
                return results;
            if (gtTok[1]=="NE_GENERIC"):
                if testTok[1][0:3]=="NE_":
                    testTok[1] = "NE_GENERIC";
            testIdx = labels.index(testTok[1]);
            confMatrix[testIdx][gtIdx] += 1;
            tot += 1;
    
    confMatrix_tr = confMatrix.transpose();
    
    corr = 0.0;
    tp = np.zeros([nLab,1]);
    tn = np.zeros([nLab,1]);
    fp = np.zeros([nLab,1]);
    fn = np.zeros([nLab,1]);
    
    p = np.zeros([nLab,1]);
    r = np.zeros([nLab,1]);
    f = np.zeros([nLab,1]);
    acc = np.zeros([nLab,1]);

    p_perclass = dict();
    r_perclass = dict();
    f_perclass = dict();
    acc_perclass = dict();

    tokens_perclass = dict();

    p_noMix = [];
    r_noMix = [];
    f_noMix = [];
    acc_noMix = [];
    tokens_noMix = 0.0;

    p_noNe = [];
    r_noNe = [];
    f_noNe = [];
    acc_noNe = [];
    tokens_noNe = 0.0;

    p_noMixAndNe = [];
    r_noMixAndNe = [];
    f_noMixAndNe = [];
    acc_noMixAndNe = [];
    tokens_noMixAndNe = 0.0;


    for ii in xrange(0, nLab):
        corr += confMatrix[ii][ii];

        tp[ii] = confMatrix[ii][ii];
        fp[ii] = confMatrix[ii].sum() - tp[ii];
        fn[ii] = confMatrix_tr[ii].sum() - tp[ii];
        tn[ii] = tot - (tp[ii]+fp[ii]+fn[ii]);
        p[ii] = tp[ii]/(tp[ii] + fp[ii]);
        if np.isnan(p[ii]):
            p[ii] = 0;
        r[ii] = tp[ii]/(tp[ii]+fn[ii]);
        if np.isnan(r[ii]):
            r[ii] = 0;
        f[ii] = 2*p[ii]*r[ii]/(p[ii]+r[ii]);
        if np.isnan(f[ii]):
            f[ii] = 0;
        acc[ii] = (tp[ii]+tn[ii])/tot;
        p_perclass[labels[ii]] = p[ii];
        r_perclass[labels[ii]] = r[ii];
        f_perclass[labels[ii]] = f[ii];
        acc_perclass[labels[ii]] = acc[ii];
        tokens_perclass[labels[ii]] = tp[ii] + fn[ii];
        isNE = False;
        isMix = False;
        if labels[ii][0:3]=="NE_":
            isNE = True;
        if labels[ii].upper()=="MIX":
            isMix = True;
        if not isNE:
            p_noNe.append(p[ii]);
            r_noNe.append(r[ii]);
            f_noNe.append(f[ii]);
            acc_noNe.append(acc[ii]);
            tokens_noNe += tp[ii] + fn[ii];
        if not isMix:
            p_noMix.append(p[ii]);
            r_noMix.append(r[ii]);
            f_noMix.append(f[ii]);
            acc_noMix.append(acc[ii]);
            tokens_noMix += tp[ii] + fn[ii];
        if (not isNE) and (not isMix):
            p_noMixAndNe.append(p[ii]);
            r_noMixAndNe.append(r[ii]);
            f_noMixAndNe.append(f[ii]);
            acc_noMixAndNe.append(acc[ii]);
            tokens_noMixAndNe += tp[ii] + fn[ii];

    results["rows"] = rcnt;
    results["tokens"] = tot;

    results["micro_accuracy"] = corr/tot;
    results["macro_accuracy"] = acc.mean();
    
    results["p_macro"] = p.mean();
    results["r_macro"] = r.mean();
    results["f_macro"] = f.mean();

    results["p_perClass"] = p_perclass;
    results["r_perClass"] = r_perclass;
    results["f_perClass"] = f_perclass;
    results["acc_perClass"] = acc_perclass;
    results["tokens_perClass"] = tokens_perclass;

    results["p_noMix"] = sum(p_noMix)/len(p_noMix);
    results["r_noMix"] = sum(r_noMix)/len(r_noMix);
    results["f_noMix"] = sum(f_noMix)/len(f_noMix);
    results["acc_noMix"] = sum(acc_noMix)/len(acc_noMix);
    results["tokens_noMix"] = tokens_noMix;

    results["p_noNe"] = sum(p_noNe)/len(p_noNe);
    results["r_noNe"] = sum(r_noNe)/len(r_noNe);
    results["f_noNe"] = sum(f_noNe)/len(f_noNe);
    results["acc_noNe"] = sum(acc_noNe)/len(acc_noNe);
    results["tokens_noNe"] = tokens_noNe;


    results["p_noMixAndNe"] = sum(p_noMixAndNe)/len(p_noMixAndNe);
    results["r_noMixAndNe"] = sum(r_noMixAndNe)/len(r_noMixAndNe);
    results["f_noMixAndNe"] = sum(f_noMixAndNe)/len(f_noMixAndNe);
    results["acc_noMixAndNe"] = sum(acc_noMixAndNe)/len(acc_noMixAndNe);
    results["tokens_noMixAndNe"] = tokens_noMixAndNe;
    
    results["message"] = "Evaluation successful";

    return results

if __name__=="__main__":
    
    gtfile = sys.argv[1];
    testfile = sys.argv[2];
    
    
    results = evaluateResult(gtfile, testfile);
    p = printResult(results);
    print p;
