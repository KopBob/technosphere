#!/usr/bin/env python
# coding: utf-8


import sys

from antispam_classifier import is_spam

DATA_FILE = sys.argv[1]

MAX_DIFF_F1 = 0.01
MIN_F1 = 0.9

summ_spam = 0
false_summ_spam = 0
detected_spam = 0
summ_not_spam = 0
false_summ_not_spam = 0
detected_not_spam = 0

with open(DATA_FILE) as df:
    for i, line in enumerate(df):
        line = line.strip()
        parts = line.split()
        # print i, int(parts[1]),
        spam_page = int(is_spam(parts[3], parts[2]))
        mark = int(parts[1])

        if mark == 1:
            summ_spam += 1
            if spam_page != mark:
                false_summ_spam += 1
            else:
                detected_spam += 1;
        else:
            summ_not_spam += 1
            if mark != spam_page:
                false_summ_not_spam += 1
            else:
                detected_not_spam += 1

SpamPrec = float(detected_spam) / (float(detected_spam) + (summ_not_spam - detected_not_spam))
SpamAccuracy = float(detected_spam) / summ_spam
print "Spam detection Accuracy:" + str(SpamAccuracy) + " Precision: " + str(SpamPrec)
F1Spam = (2 * SpamAccuracy * SpamPrec) / (SpamAccuracy + SpamPrec)
print "Spam F1 metric: " + str(F1Spam)
print
NotSpamPrec = float(detected_not_spam) / (float(detected_not_spam) + (summ_spam - detected_spam))
NotSpamAccuracy = float(summ_not_spam - false_summ_not_spam) / summ_not_spam;
print "Not Spam detection Accuracy:" + str(NotSpamAccuracy) + " Precision: " + str(NotSpamPrec)
F1NotSpam = (2 * NotSpamAccuracy * NotSpamPrec) / (NotSpamAccuracy + NotSpamPrec)
print "Not Spam F1 metric: " + str(F1NotSpam)
print
diff = 0
if F1Spam < F1NotSpam:
    diff = 1 - F1Spam / F1NotSpam
else:
    diff = 1 - F1NotSpam / F1Spam

print "F1 balance: " + str(diff)

if F1NotSpam < MIN_F1 or F1Spam < MIN_F1:
    print "F1 is too small. Min threshould:  " + str(MIN_F1) + " => fail"

if diff > MAX_DIFF_F1:
    print "F1 balance more then: " + str(MAX_DIFF_F1) + " => fail"
else:
    result = 0
    if F1NotSpam > MIN_F1 and F1Spam > MIN_F1:
        result = 6
        for i in range(1, 6):
            if F1NotSpam > MIN_F1 + (i * 0.01) and F1Spam > MIN_F1 + (i * 0.01):
                result += 1
        print "Your score: " + str(result)



# In[ ]:
