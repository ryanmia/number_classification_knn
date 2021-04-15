clear all;close all;

train_img=loadMNISTImages("train-images.idx3-ubyte");
test_img=loadMNISTImages("t10k-images.idx3-ubyte");
train_labels=loadMNISTLabels("train-labels.idx1-ubyte");
test_labels=loadMNISTLabels("t10k-labels.idx1-ubyte");

 W_train=pca(train_img');
 
W_train_5=W_train(:,1:5);
W_train_10=W_train(:,1:10);
W_train_20=W_train(:,1:20);

y_train_5=W_train_5'*train_img;
y_test_5=W_train_5'*test_img;

y_train_10=W_train_10'*train_img;
y_test_10=W_train_10'*test_img;

y_train_20=W_train_20'*train_img;
y_test_20=W_train_20'*test_img;

predicted=knn(20,y_test_5,y_train_5,train_labels);
correct_rate_knn_EIG5=get_correct_rate(predicted,test_labels)

predicted=knn(20,y_test_10,y_train_10,train_labels);
correct_rate_knn_EIG10=get_correct_rate(predicted,test_labels)

predicted=knn(20,y_test_20,y_train_20,train_labels);
correct_rate_knn_EIG20=get_correct_rate(predicted,test_labels)

train_img=train_img(:,1:20000);
train_labels=train_labels(1:20000);
test_img=test_img(:,1:10000);
test_labels=test_labels(1:10000);

W_train=pca(train_img');
 
W_train_5=W_train(:,1:5);
W_train_10=W_train(:,1:10);
W_train_20=W_train(:,1:20);

y_train_5=(W_train_5'*train_img)';
y_test_5=(W_train_5'*test_img)';

y_train_10=(W_train_10'*train_img)';
y_test_10=(W_train_10'*test_img)';

y_train_20=(W_train_20'*train_img)';
y_test_20=(W_train_20'*test_img)';

t_box1_gauss=templateSVM('Standardize',true,'KernelFunction','gaussian','BoxConstraint',1);
t_box10_gauss=templateSVM('Standardize',true,'KernelFunction','gaussian','BoxConstraint',10);
t_box100_gauss=templateSVM('Standardize',true,'KernelFunction','gaussian','BoxConstraint',100);

t_box1_poly=templateSVM('Standardize',true,'KernelFunction','polynomial','BoxConstraint',1);
t_box10_poly=templateSVM('Standardize',true,'KernelFunction','polynomial','BoxConstraint',10);
t_box100_poly=templateSVM('Standardize',true,'KernelFunction','polynomial','BoxConstraint',100);




Md1_gauss_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box1_gauss);
Md10_gauss_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box10_gauss);
Md100_gauss_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box100_gauss);

Md1_gauss_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box1_gauss);
Md10_gauss_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box10_gauss);
Md100_gauss_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box100_gauss);

Md1_poly_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box1_poly);
Md10_poly_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box10_poly);
Md100_poly_EIG5=fitcecoc(y_train_5,train_labels,'Learners',t_box100_poly);

Md1_poly_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box1_poly);
Md10_poly_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box10_poly);
Md100_poly_EIG10=fitcecoc(y_train_10,train_labels,'Learners',t_box100_poly);




predicted=predict(Md1_gauss_EIG5,y_test_5);correct_rate_svm_Md1_gauss_EIG5=get_correct_rate(predicted,test_labels)
predicted=predict(Md10_gauss_EIG5,y_test_5);correct_rate_svm_Md10_gauss_EIG5=get_correct_rate(predicted,test_labels)
predicted=predict(Md100_gauss_EIG5,y_test_5);correct_rate_svm_Md100_gauss_EIG5=get_correct_rate(predicted,test_labels)

predicted=predict(Md1_gauss_EIG10,y_test_10);correct_rate_svm_Md1_gauss_EIG10=get_correct_rate(predicted,test_labels)
predicted=predict(Md10_gauss_EIG10,y_test_10);correct_rate_svm_Md10_gauss_EIG10=get_correct_rate(predicted,test_labels)
predicted=predict(Md100_gauss_EIG10,y_test_10);correct_rate_svm_Md100_gauss_EIG10=get_correct_rate(predicted,test_labels)


predicted=predict(Md1_poly_EIG5,y_test_5);correct_rate_svm_y_test_5=get_correct_rate(predicted,test_labels)
predicted=predict(Md10_poly_EIG5,y_test_5);correct_rate_svm_Md10_poly_EIG5=get_correct_rate(predicted,test_labels)
predicted=predict(Md100_poly_EIG5,y_test_5);correct_rate_svm_Md100_poly_EIG5=get_correct_rate(predicted,test_labels)

predicted=predict(Md1_poly_EIG10,y_test_10);correct_rate_svm_Md1_poly_EIG10=get_correct_rate(predicted,test_labels)
predicted=predict(Md10_poly_EIG10,y_test_10);correct_rate_svm_Md10_poly_EIG10=get_correct_rate(predicted,test_labels)
predicted=predict(Md100_poly_EIG10,y_test_10);correct_rate_svm_Md100_poly_EIG10=get_correct_rate(predicted,test_labels)
