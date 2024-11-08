clear;
clc;
close all;
% tic;


%% Adding Path
addpath(genpath('./datasets'));
addpath(genpath('./SkewedT'));

%% Load data
[imgname,pathname]=uigetfile('*.*', 'Select the  *.mat dataset','.\datasets');
str=strcat(pathname,imgname);
disp('The dataset is :')
disp(str);
addpath(pathname);
load(strcat(pathname,imgname));

%% Reading data
mask=map;
% mask=groundtruth;
[m,n,b]=size(data);
N=m*n;
dat=normalize(data);
tic;
%% AE
n_hid1=100;             %the number of layer 1 nodes
n_hid2=50;              %the number of layer 2 nodes
n_hid3=25;              %the number of hidden layer 3 nodes
n_hid4=50;              %the number of layer 4 nodes
n_hid5=100;             %the number of layer 5 nodes
lr=0.4;                  % learning rate0.4
epchoes=300;            % Itermax300
beta=1;                 % Penalty coefficient1
batch_num=10;           % batch10
batch_size=N/batch_num; %batchsize
iter=1;


%% perform anomaly detection with CTAD calculate guided map

sci = GIC(dat);

[REC,Y3,loss] = trainmyAE(n_hid1,n_hid2,n_hid3,n_hid4,n_hid5,lr,epchoes,batch_num,batch_size,beta,sci,dat); %train
toc
figure;
plot(loss);       %loss
res=reshape(REC',m,n,b);
figure;imshow(res(:,:,10));
err2=(dat-res).^2;

% False Alarm Rate for Detection Map 
DAT.FARValue=0.05; 
FiltOn=1; % 1 - stdfilt, 2- average+ square, 3- closingfilter, 4- closingfilter + average+ square 5- No filter
%6 - Guardband Filter(only skewedt dist) 7-Guardband Filter( just skewedt dist) +stdfilt
PCAOn=0; % 1 - PCA On, 0 - PCA Off
DAT.map=mask;
DAT.D1=res;

%% Anomaly Detection for MV Skewed Distribution
[Result.AUCMeanSkewed,Result.AucModeSkewed,Result.elapsedTimeSkewed,Result.index,...
    Result.SkwFARMeanResult,Result.SkwFARModeResult,...
    Result.dtmapSkwMeanArray,Result.dtmapSkwModeArray,Result.dectionmap]=AllSkewed(DAT,FiltOn,PCAOn);

gae=mat2gray(sum(err2,3));
a=normalize(Result.dectionmap);
gae=normalize(gae);
aa=gae+a;
aa=normalize(aa);
gae1=aa;

toc;
%% 
det_map=reshape(gae1,N,1);
GT=reshape(mask,N,1);
mode_eq=1;
[AUC_D_F,AUC_D_tau,AUC_F_tau,AUC_TD,AUC_BS,AUC_SNPR,AUC_TDBS,AUC_ODP]=plot_3DROC(det_map,GT,mode_eq);

