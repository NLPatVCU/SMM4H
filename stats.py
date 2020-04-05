with open("data/train/labels") as label_file_processed:
    labels = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()

with open("data/train/labels_val") as label_file_processed:
    labels_val = [int(label.rstrip()) for label in label_file_processed]
    label_file_processed.close()


print("---------------------- Train Data ---------------------------------")
print("Total Tweets: " + str(labels.count(0)+labels.count(1)))
print("Negative: " + str(labels.count(0)))
print("Positives: " + str(labels.count(1)))
print("Percent Negative: " + str((labels.count(0)/(labels.count(1)+labels.count(0)))*100))
print("Percent Negative: "+ str((labels.count(1)/(labels.count(1)+labels.count(0)))*100))

print("---------------------- Validation Data -----------------------------")
print("Total Tweets: " + str(labels_val.count(0)+labels_val.count(1)))
print("Negative: " + str(labels_val.count(0)))
print("Positives: " + str(labels_val.count(1)))
print("Percent Negative: " + str((labels_val.count(0)/(labels_val.count(1)+labels_val.count(0)))*100))
print("Percent Negative: " + str((labels_val.count(1)/(labels_val.count(1)+labels_val.count(0)))*100))
