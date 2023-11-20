# eicza
ear dataset project for zambian infant cohort study

Update 9/28/2023:
I will be uploading the EICZA dataset shortly in these two weeks.

Update 11/20/2023:
Sorry for the long wait. The EICZA dataset w/ annotation is in [here](https://drive.google.com/drive/folders/1ER7F5brWqPFWv3oQ5HFKVGa01EYWb6p3?usp=sharing):

The images are saved in jpgs/ and 
there are two sets of annotations, "anno_group_w_age_progression.csv", and "anno_group_wo_age_progression.csv" for a 4-fold cross-validation evaluation.
The annotation format is as follows:

* idx: the unique ID for each ear image.

* split: the data is split into 4*3 = 12 sets. The 4 stands for all ear subjects are divided into 4 folds (e.g. split 0,1,2,3) so users can conduct 4-fold cross-validation conveniently. The 3 stands for 3 different sub-groups further for each ear subject (e.g., split 0,4,8 for ear subject "6D41"). In "anno_group_w_age_progression.csv", the smaller the split number is, the younger the ears are assigned. For example, for ear subject "6D41", split 0 contains his/her ears captured at age 6 days (6D), split 4 contains captured ears at his/her 6 weeks and split 8 is for his/her 10-week-old ears. This allows a quick organization to train a model with young ears to predict older ears for the same subject. For "anno_group_w_age_progression.csv", the age of the ears is randomly shuffled, as a comparison with the organized one.

* earSubjectIdx: a unique ID for each ear subject.

* earImageName: the image file name for a specific ear image in folder jpg/. So you can access the ear data in the annotations from the folder by the file name.

* Period: the age group of the captured ear image, there are 10 different groups: 6-day (6D), 6-week (6W), 10-week (10W), 14-week (14W), 4-month (4M), 5-month (5M), 6-month (6M), 7-month (7M), 8-month (8M) and 9-* month (9M). For example, 6M means the ear image is captured when the ear subject is 6 months old.

* earSubjectName: another unique ID for the ear subject, they should have a 1-to-1 matching with earSubjectIdx.

* earLeftRight: whether the ear image belongs to a left ear or right ear.

* earRotated: whether the ear image is taken with a 30-degree rotation regarding the camera device.


As soon as I get the time, I will also update the code implemented for EICZA benchmarking and the SASE model. Sorry again for the wait.
