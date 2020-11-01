clc;
clear all;

%% Read Data
X = readtable('Sensorless_drive_diagnosis.txt','ReadVariableNames',false);
Data = table2array(X);
RandData = Data(randperm(size(Data, 1)),:);
t = Data(:,end);
class1 = Data(t==1,:); class2 = Data(t==2,:); class3 = Data(t==3,:); class4 = Data(t==4,:);...
    class5 = Data(t==5,:); class6 = Data(t==6,:); class7 = Data(t==7,:); class8 = Data(t==8,:);...
    class9 = Data(t==9,:); class10 = Data(t==10,:); class11 = Data(t==11,:);
L = length(t);
TrainDataLen = round(length(class1)*0.2);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
% Feature Selectiion: Using Visual Inspection from Histogram Plot
%NumFeature = [1:48];
%NumFeature = [1:7,10,13:19,25:34, 37:43,46];
NumFeature = [1:7,10,13,14,15,18,19,25:31,34 37:43,46];
figure(1);
for i = NumFeature
    subplot(6,8,i)
    histogram(class11(:,i));
    xlabel(i)
end

%Feature = RandData(:,NumFeature); %Feature set 1
Target = RandData(:,end);

% Feature Selectiion: Using Relieff Function
[Predictor_ranks,Predictor_weights] = relieff(RandData(:,1:48),RandData(:,end),10);
Feature = RandData(:,Predictor_ranks(1:30)); %Feature set 2

%% Dimentionality Reduction
Data2 = normalize(RandData(:,1:48),'range',[0,1]);
[Coeff, NewData, EigenValues, TSquared, Variance] = pca(Data2);

for m = 1:48
    Percent_variance(m) = sum(Variance(1:m));
end

figure(2), 
plot(1:length(Percent_variance),Percent_variance,'LineWidth',2);
hold on;
plot(100*ones(1,48),'r');
hold on;
plot([zeros(1,19),100,zeros(1,28)],'r');
%text(1:length(Percent_variance),Percent_variance,num2str(Percent_variance'),'vert','bottom','horiz','center'); 
ylim([70 105]);
box off;
xlabel('Features'); 
ylabel('Percentage Variance');
%title('Sum of percentage variance of features after PCA, we can see that first 29 features carry the 100 variance of the data.');

%Feature = NewData(:,1:29); %Feature set 3
%% %%%%%%%%%%%%%%%%%%%%%%% Fitting Multivariate Gaussian to Data %%%%%%%%%%%%%%%%%%%%%%%% %%
% Fitting MVNPDF to Data
for N = 532:532:4*1064 % Choice of Size of training data set

for k = 1:1 % Randomized trainning data set over 1000 times
    
%index = randperm(5319)';
index  = 1:5319;
% Training Data
T1 = class1(index(1:N),NumFeature); T2 = class2(index(1:N),NumFeature); T3 = class2(index(1:N),NumFeature);
T4 = class4(index(1:N),NumFeature); T5 = class5(index(1:N),NumFeature); T6 = class6(index(1:N),NumFeature);
T7 = class7(index(1:N),NumFeature); T8 = class8(index(1:N),NumFeature); T9 = class9(index(1:N),NumFeature);
T10 = class10(index(1:N),NumFeature); T11 = class11(index(1:N),NumFeature);

% Gaussian parameters
% mu 
mT1 = mean(class1(:,NumFeature)); mT2 = mean(class2(:,NumFeature)); mT3 = mean(class3(:,NumFeature)); mT4 = mean(class4(:,NumFeature));
mT5 = mean(class5(:,NumFeature)); mT6 = mean(class6(:,NumFeature)); mT7 = mean(class7(:,NumFeature)); mT8 = mean(class8(:,NumFeature));
mT9 = mean(class9(:,NumFeature)); mT10 = mean(class10(:,NumFeature)); mT11 = mean(class11(:,NumFeature));


% Sigma
sT1 =  ((T1-repmat(mT1,length(T1),1))'*(T1-repmat(mT1,length(T1),1)) ) /N;
sT2 =  ((T2-repmat(mT2,length(T2),1))'*(T2-repmat(mT2,length(T2),1)) ) /N;
sT3 =  ((T3-repmat(mT3,length(T3),1))'*(T3-repmat(mT3,length(T3),1)) ) /N;
sT4 =  ((T4-repmat(mT4,length(T4),1))'*(T4-repmat(mT4,length(T4),1)) ) /N;
sT5 =  ((T5-repmat(mT5,length(T5),1))'*(T5-repmat(mT5,length(T5),1)) ) /N;
sT6 =  ((T6-repmat(mT6,length(T6),1))'*(T6-repmat(mT6,length(T6),1)) ) /N;
sT7 =  ((T7-repmat(mT7,length(T7),1))'*(T7-repmat(mT7,length(T7),1)) ) /N;
sT8 =  ((T8-repmat(mT8,length(T8),1))'*(T8-repmat(mT8,length(T8),1)) ) /N;
sT9 =  ((T9-repmat(mT9,length(T9),1))'*(T9-repmat(mT9,length(T9),1)) ) /N;
sT10 =  ((T10-repmat(mT10,length(T10),1))'*(T10-repmat(mT10,length(T10),1)) ) /N;
sT11 =  ((T11-repmat(mT11,length(T11),1))'*(T11-repmat(mT11,length(T11),1)) ) /N;


% Test data
Testdata = [class1(index(N+1:end),NumFeature);class2(index(N+1:end),NumFeature);...
    class3(index(N+1:end),NumFeature);class4(index(N+1:end),NumFeature);...
    class5(index(N+1:end),NumFeature);class6(index(N+1:end),NumFeature);...
    class7(index(N+1:end),NumFeature);class8(index(N+1:end),NumFeature);...
    class9(index(N+1:end),NumFeature);class10(index(N+1:end),NumFeature);...
    class11(index(N+1:end),NumFeature)];

%Maximum Likelihood
P1 = [mvnpdf(Testdata(:,:),mT1,sT1),mvnpdf(Testdata(:,:),mT2,sT2),mvnpdf(Testdata(:,:),mT3,sT3),...
    mvnpdf(Testdata(:,:),mT4,sT4),mvnpdf(Testdata(:,:),mT5,sT5),mvnpdf(Testdata(:,:),mT6,sT6),...
    mvnpdf(Testdata(:,:),mT7,sT7),mvnpdf(Testdata(:,:),mT8,sT8),mvnpdf(Testdata(:,:),mT9,sT9),...
    mvnpdf(Testdata(:,:),mT10,sT10),mvnpdf(Testdata(:,:),mT11,sT11)];
[P1max,Class] = max(P1,[],2);
TrueClass = [ones(length(class1)-N,1);2*ones(length(class2)-N,1);3*ones(length(class3)-N,1);...
    4*ones(length(class1)-N,1);5*ones(length(class2)-N,1);6*ones(length(class3)-N,1);...
    7*ones(length(class1)-N,1);8*ones(length(class2)-N,1);9*ones(length(class3)-N,1);...
    10*ones(length(class1)-N,1);11*ones(length(class2)-N,1)];

Count = 0;
for i = 1:length(Testdata)
    if Class(i) == TrueClass(i)
        Count = Count + 1;
    end
end
misclassification_Rate(k) = (length(Testdata)-Count)/length(Testdata);
end
Avg_Miss_Class_Rate(N/532) = mean(misclassification_Rate);
end

% Misclassification Rate Plot
figure(3);
plot([1*532, 2*532, 3*532, 4*532, 5*532, 6*532, 7*532, 8*532],[0.776060162941299,0.638564255955560,0.465191805240153,0.389703997037121,0.347704195015214,0.302517416762833,0.259675121117127,0.242965877020440],'b','LineWidth',2);
hold on;
plot([1*532, 2*532, 3*532, 4*532, 5*532, 6*532, 7*532, 8*532],[0.770989612017396,0.629740412349108,0.454960564549606,0.381442124156007,0.345413518410886,0.302431935718255,0.269307495012824,0.253827076028393],'g','LineWidth',2);
hold on;
plot([1*532, 2*532, 3*532, 4*532, 5*532, 6*532, 7*532, 8*532],Avg_Miss_Class_Rate,'r','LineWidth',2);
xlabel('Size of the training data')
ylabel('% Misclassification rate')
legend('All Features','Feature Set-1','Feature Set-2');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Naive Bayes
err_NvBayes = 0;
Model_NvBayes = fitcnb(Feature,Target,'CrossVal','on');
Y_NvBayes = kfoldPredict(Model_NvBayes);
err_NvBayes = err_NvBayes + sum(Y_NvBayes~=Target)/L;
ConfMat_NvBayes = confusion.getMatrix(Target,Y_NvBayes,0);
figure(5)
plotConfMat(ConfMat_NvBayes)
Accuracy_NvBayes = 100*trace(ConfMat_NvBayes)/sum(ConfMat_NvBayes(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('Naive Bayes Accuracy: %.2f%%',Accuracy_NvBayes);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LDA
%Feature = normalize(Feature,'range',[0,1]);
err_lda = 0;
Model_LDA = fitcdiscr(Feature,Target,'CrossVal','on');
Y_LDA = kfoldPredict(Model_LDA);
err_lda = err_lda + sum(Y_LDA~=Target)/L;
ConfMat_LDA = confusion.getMatrix(Target,Y_LDA,0);
figure(6)
plotConfMat(ConfMat_LDA);
Accuracy_LDA = 100*trace(ConfMat_LDA)/sum(ConfMat_LDA(:));
xlabel('True Class'); ylabel('Predicted Class'); %title('LDA Accuracy: %.2f%%', Accuracy_LDA);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%QDA
err_qda = 0;
Model_QDA = fitcdiscr(Feature,Target,'CrossVal','on','DiscrimType','quadratic');
Y_QDA = kfoldPredict(Model_QDA);
err_qda = err_qda + sum(Y_QDA~=Target)/L;
ConfMat_QDA = confusion.getMatrix(Target,Y_QDA,0);
figure(7)
plotConfMat(ConfMat_QDA);
Accuracy_QDA = 100*trace(ConfMat_QDA)/sum(ConfMat_QDA(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('QDA Accuracy: %.2f%%',Accuracy_QDA);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%KNN
k = 10;
    %KNN euclidian distance
t1 = templateKNN('NumNeighbors',k,'Distance','euclidean','Standardize',1);
Model1 = fitcecoc(Feature,Target,'Learners',t1,'crossval','on'); 
kNNmisClassRate1 = Model1.kfoldLoss;
Y_KNN1 = kfoldPredict(Model1);
err_knn1 = sum(Y_KNN1~=Target)/L;

ConfMat_KNN1 = confusion.getMatrix(Target,Y_KNN1,0);
figure(8)
plotConfMat(ConfMat_KNN1)
Accuracy_KNN1 = 100*trace(ConfMat_KNN1)/sum(ConfMat_KNN1(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('KNN euclidian Accuracy: %.2f%%',Accuracy_KNN1);

    %KNN cosine distance
t2 = templateKNN('NumNeighbors',k,'Distance','cosine','Standardize',1);
Model2 = fitcecoc(Feature,Target,'Learners',t2,'crossval','on'); 
kNNmisClassRate2 = Model2.kfoldLoss;
Y_KNN2 = kfoldPredict(Model2);
err_knn2 = sum(Y_KNN2~=Target)/L;

ConfMat_KNN2 = confusion.getMatrix(Target,Y_KNN2,0);
figure(9)
plotConfMat(ConfMat_KNN2);
Accuracy_KNN2 = 100*trace(ConfMat_KNN2)/sum(ConfMat_KNN2(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('KNN cosine Accuracy: %.2f%%',Accuracy_KNN2);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SVM
%SVM: Linear
t_SVM1 = templateSVM('BoxConstraint',1,'KernelFunction','linear','KernelScale','auto','Standardize',1)
Model_SVM1 = fitcecoc(Feature,Target,'Learners',t_SVM1,'crossval','on'); SVMmisClassRate1 = Model_SVM1.kfoldLoss;
Y_SVM1 = kfoldPredict(Model_SVM1);

ConfMat_SVM1 = confusion.getMatrix(Target,Y_SVM1,0);
figure(10)
plotConfMat(ConfMat_SVM1)
Accuracy_SVM1 = 100*trace(ConfMat_SVM1)/sum(ConfMat_SVM1(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('SVM Linear Accuracy: %.2f%%',Accuracy_SVM1);

%SVM: Gaussian
t_SVM2 = templateSVM('BoxConstraint',1,'KernelFunction','gaussian','KernelScale',6.9,'Standardize',1)
Model_SVM2 = fitcecoc(Feature,Target,'Learners',t_SVM2,'crossval','on'); SVMmisClassRate2 = Model_SVM2.kfoldLoss;
Y_SVM2 = kfoldPredict(Model_SVM2);

ConfMat_SVM2 = confusion.getMatrix(Target,Y_SVM2,0);
figure(11)
plotConfMat(ConfMat_SVM2)
Accuracy_SVM2 = 100*trace(ConfMat_SVM2)/sum(ConfMat_SVM2(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('SVM Gaussian Accuracy: %.2f%%',Accuracy_SVM2);

%SVM: Polynomial
% t_SVM3 = templateSVM('BoxConstraint',1,'KernelFunction','polynomial','PolynomialOrder',10,'KernelScale','auto','Standardize',1)
% Model_SVM3 = fitcecoc(Feature,Target,'Learners',t_SVM3,'crossval','on'); SVMmisClassRate3 = Model_SVM3.kfoldLoss;
% Y_SVM3 = kfoldPredict(Model_SVM3);
% 
% ConfMat_SVM3 = confusion.getMatrix(Target,Y_SVM3,0);
% figure(12)
% plotConfMat(ConfMat_SVM3)
% Accuracy_SVM3 = 100*trace(ConfMat_SVM3)/sum(ConfMat_SVM3(:));
% xlabel('True Class'), ylabel('Predicted Class'), title('SVM Polynomial Accuracy: %.2f%%',Accuracy_SVM3);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Decission Tree
Model_DT = fitcecoc(Feature,Target,'Learners','tree','crossval','on'); 
TREEmisClassRate = Model_DT.kfoldLoss;
Y_DT = kfoldPredict(Model_DT);

ConfMat_DT = confusion.getMatrix(Target,Y_DT,0);
figure(12)
plotConfMat(ConfMat_DT);
Accuracy_DT = 100*trace(ConfMat_DT)/sum(ConfMat_DT(:));
xlabel('True Class'), ylabel('Predicted Class'), %title('Decission Tree Accuracy: %.2f%%',Accuracy_DT);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%comparison of all classification algorithm
figure(13)
Rate1 = [58.96,83.39,83.35,94,93.38,92.65,96.90,98.46];
Rate2 = [78.74,82.89,84.39,98.91,98.62,91.73,98.53,98.51];
Rate3 = [72.27,84.75,84.29,90.28,89.06,92.56,95.95,97.51];
c = categorical({'Naive Bayes','LDA','QDA','KNN euclidean','KNN Cosine','SVM Linear','SVM Gaussian','Decision Tree'});
bar(c,[Rate1;Rate2;Rate3]');
legend('Feature Set 1','Feature Set 2','Feature Set 2');
%text(1:length(Rate1),Rate1,num2str(Rate1'),'vert','bottom','horiz','center');
ylim([0 110]);
box off
%grid on;
ylabel('% Accuracy');
xlabel('Algorithm Type');



